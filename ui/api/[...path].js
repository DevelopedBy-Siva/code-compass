import {
  InvokeEndpointCommand,
  SageMakerRuntimeClient,
} from "@aws-sdk/client-sagemaker-runtime";
import { awsCredentialsProvider } from "@vercel/oidc-aws-credentials-provider";

const decoder = new TextDecoder();

function getPathSegments(req) {
  const queryPath = req.query.path;
  const fromQuery = Array.isArray(queryPath)
    ? queryPath
    : typeof queryPath === "string"
      ? queryPath.split("/")
      : [];

  const querySegments = fromQuery
    .flatMap((part) => String(part).split("/"))
    .filter(Boolean);
  if (querySegments.length > 0) {
    return querySegments;
  }

  const requestPath = new URL(req.url, "http://localhost").pathname;
  return requestPath
    .replace(/^\/api\/?/, "")
    .split("/")
    .filter(Boolean);
}

function getClient() {
  const roleArn = process.env.AWS_ROLE_ARN;
  const region = process.env.SAGEMAKER_AWS_REGION || process.env.AWS_REGION;
  if (!roleArn || !region) {
    throw new Error("AWS_ROLE_ARN and SAGEMAKER_AWS_REGION are required");
  }
  return new SageMakerRuntimeClient({
    region,
    credentials: awsCredentialsProvider({
      roleArn,
      clientConfig: { region },
      roleSessionName: "code-compass-vercel",
    }),
  });
}

function mapRequest(req) {
  const path = getPathSegments(req);
  const sessionId = req.headers["x-session-id"] || req.query.session_id;
  if (!sessionId) {
    return { error: [400, "Missing session id"] };
  }

  if (req.method === "GET" && path.join("/") === "repos") {
    return { action: "list_repositories", session_id: sessionId, payload: {} };
  }
  if (req.method === "GET" && path[0] === "repos" && path[1]) {
    return {
      action: "get_repository",
      session_id: sessionId,
      payload: { repo_id: Number(path[1]) },
    };
  }
  if (req.method === "POST" && path.join("/") === "repos/index") {
    return { action: "index_repository", session_id: sessionId, payload: req.body };
  }
  if (req.method === "POST" && path.join("/") === "query") {
    return { action: "query", session_id: sessionId, payload: req.body };
  }
  if (req.method === "POST" && path.join("/") === "session/end") {
    return { action: "end_session", session_id: sessionId, payload: {} };
  }
  return { error: [404, "Route not found"] };
}

export default async function handler(req, res) {
  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  const invocation = mapRequest(req);
  if (invocation.error) {
    return res.status(invocation.error[0]).json({ detail: invocation.error[1] });
  }

  try {
    const endpointName = process.env.SAGEMAKER_ENDPOINT_NAME;
    if (!endpointName) {
      throw new Error("SAGEMAKER_ENDPOINT_NAME is required");
    }
    const response = await getClient().send(
      new InvokeEndpointCommand({
        EndpointName: endpointName,
        ContentType: "application/json",
        Accept: "application/json",
        Body: JSON.stringify(invocation),
      }),
    );
    const body = decoder.decode(response.Body);
    return res.status(200).json(body ? JSON.parse(body) : {});
  } catch (error) {
    console.error("SageMaker invocation failed", error);
    const status = Number(
      error?.OriginalStatusCode || error?.originalStatusCode || 502,
    );
    let detail = "Backend invocation failed";
    const originalMessage = error?.OriginalMessage || error?.originalMessage;
    if (status < 500 && originalMessage) {
      try {
        detail = JSON.parse(originalMessage)?.detail || detail;
      } catch {
        detail = originalMessage;
      }
    }
    return res.status(status >= 400 && status < 600 ? status : 502).json({
      detail,
    });
  }
}
