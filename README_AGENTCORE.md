# Shipping the Data Scientist crew on Amazon Bedrock AgentCore

This converts the FastAPI/WebSocket app into an **AgentCore Runtime** agent. The
CrewAI pipeline (`crew_ds.py`) is unchanged; `agentcore_app.py` is the new
entrypoint. Datasets come in from S3 and every artifact goes back out to S3.

```
agentcore_app.py   AgentCore entrypoint (start / answer / stop / presign_upload, streaming, S3)
crew_ds.py         the crew (unchanged except WORKSPACE_ROOT is now env-driven)
s3_store.py        dataset download + artifact upload/presign + upload presign
requirements.txt   runtime dependencies
static/index.html  browser UI — Cognito login + streaming invoke + S3 upload (§7)
main.py            OLD FastAPI server — not deployed; kept for local reference only
```

## How human-in-the-loop works here

AgentCore is request/response, but a runtime **session keeps the same microVM
warm across invocations for up to 8h**. So the crew runs in a background thread;
when it calls `ask_user` the thread blocks and the invocation returns a
`question` event. You reply with a second invocation (`action:"answer"`) **using
the same session id**, which unblocks the thread and resumes streaming. If the
session idles out (~15 min) the paused run is lost — start again.

---

## 1. Prerequisites

- **Bedrock model access** enabled in your region for the models you use
  (default `global.amazon.nova-2-lite-v1:0` and vision `qwen.qwen3-vl-235b-a22b`).
- **An S3 bucket** for datasets + artifacts, e.g. `my-ds-agent-bucket`.
- **Docker** running locally (the toolkit builds an ARM64 image), or rely on the
  managed CodeBuild path `agentcore launch` uses.
- Python 3.11+ and AWS credentials configured (`aws sts get-caller-identity`).

```bash
pip install bedrock-agentcore bedrock-agentcore-starter-toolkit
```

## 2. IAM execution role

The runtime needs an execution role. Let the toolkit create one
(`--create-execution-role` below) or attach a role whose policy allows at least:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "*" },
    { "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::my-ds-agent-bucket/*" },
    { "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::my-ds-agent-bucket/ds-agent/*" }
  ]
}
```
(CloudWatch Logs / ECR / CodeBuild permissions are added by the toolkit when it
creates the role for you.)

## 3. Configure

Run from this directory (it must contain `agentcore_app.py` + `requirements.txt`).
`configure` takes **no** `--env` — env vars are set on `launch` (step 4). Omit
the execution role and the toolkit creates one for you (add `-ni` to skip
prompts); pass `-er <role-arn>` only if you already have one.

```bash
agentcore configure -e agentcore_app.py -rf requirements.txt
```

## 4. Launch

Env vars are passed here, one `--env KEY=VALUE` each. `WORKSPACE_ROOT=/tmp/workspace`
is important — `/tmp` is the writable path in the runtime microVM.

```bash
agentcore launch \
  --env AWS_REGION=ap-south-1 \
  --env WORKSPACE_ROOT=/tmp/workspace \
  --env ARTIFACT_BUCKET=my-ds-agent-bucket \
  --env ARTIFACT_PREFIX=ds-agent \
  --env DEFAULT_MODEL=bedrock/global.amazon.nova-2-lite-v1:0 \
  --env VISION_MODEL=bedrock/qwen.qwen3-vl-235b-a22b
```
Builds the ARM64 image, pushes to ECR, and deploys. Note the returned
**Agent Runtime ARN**.

## 5. Invoke

**Start a run** (dataset already in S3):

```bash
agentcore invoke \
  --session-id session-0001-abcdefghij-1234567890 \
  '{"action":"start",
    "prompt":"Predict Price. Regression, RMSE. Do not ask me questions, you decide.",
    "dataset_s3":"s3://my-ds-agent-bucket/uploads/ToyotaCorolla.csv",
    "mode":"autopilot"}'
```
The response streams `info/stdout/stage/answer/...` events, then `artifacts`
(presigned URLs) and `done`.

**If you get a `question` event**, reply on the **same session id**:

```bash
agentcore invoke --session-id session-0001-abcdefghij-1234567890 \
  '{"action":"answer","content":"Use median imputation."}'
```

**Follow-up after the pipeline finished** (same session id, artifacts persist in
the microVM for the session lifetime):

```bash
agentcore invoke --session-id session-0001-abcdefghij-1234567890 \
  '{"action":"start","mode":"followup","prompt":"Also try a GradientBoosting model and compare."}'
```

> Session ids must be **33+ characters**. Use a UUID-ish value.

## 6. Local smoke test (no deploy)

```bash
WORKSPACE_ROOT=/tmp/ws ARTIFACT_BUCKET= python agentcore_app.py   # serves :8080
curl -N -XPOST localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"action":"start","prompt":"hello","mode":"copilot"}'
```

---

## 7. Wiring the S3-hosted frontend (Cognito + CloudFront, no server)

`static/index.html` is already converted: it signs in with **Cognito**, calls
`InvokeAgentRuntime` with a **JWT bearer token**, streams the SSE response, and
uploads datasets with an **agent-issued presigned PUT** (so the browser holds no
AWS credentials — only the Cognito token).

### 7a. The CORS reality (why CloudFront)

The runtime data-plane endpoint (`bedrock-agentcore.<region>.amazonaws.com`) does
**not** send CORS headers, so a browser `fetch()` to it straight from your S3
page is blocked. Fix: front everything with **one CloudFront distribution** so the
page and the agent share an origin — then CORS never applies.

```
CloudFront distribution
├─ default behavior   /*             → Origin: S3 bucket (static site)  → serves index.html
└─ behavior           /runtimes/*    → Origin: bedrock-agentcore.<region>.amazonaws.com
                                        · Origin protocol: HTTPS only
                                        · Forward header: Authorization
                                        · Forward header: X-Amzn-Bedrock-AgentCore-Runtime-Session-Id
                                        · Cache policy: CachingDisabled
                                        · Origin request policy: forward all viewer headers/query strings
```
In `index.html`, leave `CONFIG.AGENT_BASE=''` (same-origin). The page then POSTs to
`https://<cloudfront-domain>/runtimes/<arn>/invocations` and CloudFront relays it
to the runtime. Serve `index.html` through CloudFront, **not** the S3 website URL.

### 7b. Cognito user pool + JWT authorizer

Create a user pool with a public app client (USER_PASSWORD_AUTH) and a user:

```bash
export REGION=ap-south-1 USERNAME=analyst PASSWORD='Str0ng-Passw0rd!'

POOL_ID=$(aws cognito-idp create-user-pool --pool-name DsAgentPool \
  --policies '{"PasswordPolicy":{"MinimumLength":8}}' --region $REGION \
  --query 'UserPool.Id' --output text)

CLIENT_ID=$(aws cognito-idp create-user-pool-client --user-pool-id $POOL_ID \
  --client-name DsAgentClient --no-generate-secret \
  --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH \
  --region $REGION --query 'UserPoolClient.ClientId' --output text)

aws cognito-idp admin-create-user --user-pool-id $POOL_ID --username $USERNAME \
  --message-action SUPPRESS --region $REGION
aws cognito-idp admin-set-user-password --user-pool-id $POOL_ID --username $USERNAME \
  --password "$PASSWORD" --permanent --region $REGION

echo "DISCOVERY_URL=https://cognito-idp.$REGION.amazonaws.com/$POOL_ID/.well-known/openid-configuration"
echo "CLIENT_ID=$CLIENT_ID"
```

Configure the runtime to accept those tokens (`-ac` takes JSON). **Note:** a
runtime is either SigV4 **or** JWT, not both — this replaces IAM inbound auth.

```bash
agentcore configure -e agentcore_app.py -rf requirements.txt \
  -ac '{"customJWTAuthorizer":{
          "discoveryUrl":"https://cognito-idp.ap-south-1.amazonaws.com/POOL_ID/.well-known/openid-configuration",
          "allowedClients":["CLIENT_ID"]}}'
agentcore launch --env AWS_REGION=ap-south-1 --env WORKSPACE_ROOT=/tmp/workspace \
  --env ARTIFACT_BUCKET=my-ds-agent-bucket --env ARTIFACT_PREFIX=ds-agent \
  --env DEFAULT_MODEL=bedrock/global.amazon.nova-2-lite-v1:0 \
  --env VISION_MODEL=bedrock/qwen.qwen3-vl-235b-a22b
```

### 7c. Fill in the page config

Edit the `CONFIG` block near the top of `static/index.html`:

```js
const CONFIG = {
  AGENT_ARN:         '<arn from agentcore launch>',
  AGENT_BASE:        '',                 // '' = same-origin via CloudFront
  COGNITO_REGION:    'ap-south-1',
  COGNITO_CLIENT_ID: '<CLIENT_ID>',
};
```

### 7d. S3 bucket CORS (uploads + reading report/notebook)

The presigned **PUT** (upload) and the `fetch()` of `report.md`/`.ipynb` go to the
S3 REST endpoint directly, so the bucket needs CORS for your CloudFront origin
(`<img>` plot loads don't need it, but PUT and text fetch do):

```json
[
  { "AllowedOrigins": ["https://YOUR_CLOUDFRONT_DOMAIN"],
    "AllowedMethods": ["GET", "PUT", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders":  ["ETag"],
    "MaxAgeSeconds":  3000 }
]
```

The runtime **execution role** also needs `s3:PutObject` on
`arn:aws:s3:::my-ds-agent-bucket/ds-agent/*` (covers both `uploads/` and
artifacts) — already in the step-2 policy.

### 7e. Flow recap

1. User signs in → Cognito access token (bearer).
2. Drop a file → page calls `{"action":"presign_upload"}` → gets a presigned PUT →
   PUTs the file to `s3://…/ds-agent/<session>/uploads/<file>`.
3. Send → `{"action":"start", dataset_s3, prompt, mode}` streams events.
4. A `question` event pauses; replying sends `{"action":"answer"}` on the **same**
   session id (the page keeps `sessionId` stable per tab).
5. `artifacts` event → plots/report/notebook render from presigned URLs; `done` ends.

## Notes & limits

- **Cold imports**: CrewAI is imported lazily inside the handler on purpose
  (it exceeds the init timeout at module load). First invocation is slower.
- **Runtime pip installs** still work as a fallback but are slow/uncertain in the
  sandbox — prefer adding libraries to `requirements.txt`.
- **Artifacts** are uploaded on `done`; `_run_*.py` scratch and the raw notebook
  log are skipped, the exported `pipeline.ipynb` is included.
