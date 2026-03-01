import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import QRCode from "qrcode";

const DEFAULT_ACTIVE_FILE = "src/main.py";
const DEFAULT_CODE = [
  "def main() -> None:",
  "    message = \"Slingshot collaborative coding workspace is ready.\"",
  "    print(message)",
  "",
  "",
  "if __name__ == \"__main__\":",
  "    main()",
].join("\n");

const DEFAULT_SCRIPT = DEFAULT_CODE;

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload && payload.detail) {
        detail = `${detail}: ${payload.detail}`;
      }
    } catch {
      // Ignore non-JSON responses.
    }
    throw new Error(detail);
  }
  return response.json();
}

function makeClientId() {
  return `client-${Math.random().toString(16).slice(2, 8)}`;
}

function normalizeWorkspacePath(path) {
  if (typeof path !== "string") {
    return "";
  }
  const candidate = path.replaceAll("\\", "/").trim().replace(/^\/+|\/+$/g, "");
  if (!candidate) {
    return "";
  }

  const parts = candidate.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) {
    return "";
  }
  return parts.join("/");
}

function extensionToLanguage(path) {
  const lower = String(path || "").toLowerCase();
  if (lower.endsWith(".py")) return "python";
  if (lower.endsWith(".js") || lower.endsWith(".mjs") || lower.endsWith(".cjs")) return "javascript";
  if (lower.endsWith(".ts") || lower.endsWith(".tsx")) return "typescript";
  if (lower.endsWith(".json")) return "json";
  if (lower.endsWith(".md")) return "markdown";
  if (lower.endsWith(".yml") || lower.endsWith(".yaml")) return "yaml";
  if (lower.endsWith(".cpp") || lower.endsWith(".cc") || lower.endsWith(".cxx")) return "cpp";
  if (lower.endsWith(".c")) return "c";
  if (lower.endsWith(".rs")) return "rust";
  if (lower.endsWith(".go")) return "go";
  if (lower.endsWith(".html")) return "html";
  if (lower.endsWith(".css")) return "css";
  return "plaintext";
}

function normalizeWorkspace(workspacePayload) {
  const now = new Date().toISOString();
  const files = {};
  const rawFiles = workspacePayload && typeof workspacePayload === "object" ? workspacePayload.files : null;
  if (rawFiles && typeof rawFiles === "object") {
    for (const [rawPath, rawEntry] of Object.entries(rawFiles)) {
      const path = normalizeWorkspacePath(rawPath);
      if (!path) {
        continue;
      }

      let content = "";
      let updatedAt = now;
      if (rawEntry && typeof rawEntry === "object") {
        content = String(rawEntry.content || "");
        updatedAt = String(rawEntry.updated_at || now);
      } else {
        content = String(rawEntry || "");
      }
      files[path] = {
        content,
        updated_at: updatedAt,
        language: extensionToLanguage(path),
      };
    }
  }

  if (!Object.keys(files).length) {
    files[DEFAULT_ACTIVE_FILE] = {
      content: DEFAULT_CODE,
      updated_at: now,
      language: extensionToLanguage(DEFAULT_ACTIVE_FILE),
    };
  }

  const requestedActive = normalizeWorkspacePath(
    workspacePayload && typeof workspacePayload === "object" ? workspacePayload.active_file : ""
  );
  const filePaths = Object.keys(files).sort();
  const activeFile = requestedActive && files[requestedActive] ? requestedActive : filePaths[0];

  return {
    files,
    active_file: activeFile,
    updated_at:
      workspacePayload && typeof workspacePayload === "object" && workspacePayload.updated_at
        ? String(workspacePayload.updated_at)
        : now,
  };
}

function buildExplorerTree(filesMap) {
  const root = { name: "", path: "", type: "folder", children: [] };
  const paths = Object.keys(filesMap || {}).sort((left, right) => left.localeCompare(right));
  const folderIndex = new Map();
  folderIndex.set("", root);

  for (const path of paths) {
    const segments = path.split("/");
    let parentPath = "";
    for (let index = 0; index < segments.length; index += 1) {
      const name = segments[index];
      const currentPath = parentPath ? `${parentPath}/${name}` : name;
      const isFile = index === segments.length - 1;

      if (isFile) {
        const parent = folderIndex.get(parentPath);
        parent.children.push({
          name,
          path: currentPath,
          type: "file",
        });
      } else {
        if (!folderIndex.has(currentPath)) {
          const folderNode = {
            name,
            path: currentPath,
            type: "folder",
            children: [],
          };
          const parent = folderIndex.get(parentPath);
          parent.children.push(folderNode);
          folderIndex.set(currentPath, folderNode);
        }
      }

      parentPath = currentPath;
    }
  }

  const sortNode = (node) => {
    if (!Array.isArray(node.children)) {
      return;
    }
    node.children.sort((left, right) => {
      if (left.type !== right.type) {
        return left.type === "folder" ? -1 : 1;
      }
      return left.name.localeCompare(right.name);
    });
    for (const child of node.children) {
      sortNode(child);
    }
  };
  sortNode(root);
  return root;
}

function inferAssetFolderName(asset) {
  const url = typeof asset?.url === "string" ? asset.url.split("?")[0] : "";
  const parts = url.split("/").filter(Boolean);
  if (parts.length >= 4 && parts[0] === "assets") {
    return parts[2];
  }

  const metadata = asset?.metadata && typeof asset.metadata === "object" ? asset.metadata : {};
  if (typeof metadata.run_id === "string" && metadata.run_id) {
    return metadata.run_id;
  }
  return "ungrouped";
}

function inferRunKind(folderName, items) {
  for (const item of items) {
    const metadata = item?.metadata && typeof item.metadata === "object" ? item.metadata : {};
    if (typeof metadata.job_kind === "string" && metadata.job_kind) {
      return metadata.job_kind.replaceAll("_", " ");
    }
  }
  if (String(folderName).startsWith("finetune_")) {
    return "fine tune";
  }
  if (String(folderName).startsWith("run_")) {
    return "pipeline";
  }
  return "mixed";
}

function groupAssetsByFolder(assetList) {
  if (!Array.isArray(assetList) || !assetList.length) {
    return [];
  }

  const groups = [];
  const indexByFolder = new Map();
  for (const asset of assetList) {
    const folder = inferAssetFolderName(asset);
    if (!indexByFolder.has(folder)) {
      indexByFolder.set(folder, groups.length);
      groups.push({
        folder,
        kind: "mixed",
        run_id: null,
        asset_count: 0,
        job: null,
        assets: [],
      });
    }

    const group = groups[indexByFolder.get(folder)];
    group.assets.push(asset);
    group.asset_count = group.assets.length;

    const metadata = asset?.metadata && typeof asset.metadata === "object" ? asset.metadata : {};
    if (!group.run_id && typeof metadata.run_id === "string" && metadata.run_id) {
      group.run_id = metadata.run_id;
    }
  }

  for (const group of groups) {
    group.kind = inferRunKind(group.folder, group.assets);
  }

  return groups;
}

export default function App() {
  const clientId = useMemo(makeClientId, []);
  const [projectId, setProjectId] = useState("");
  const [workspace, setWorkspace] = useState(() => normalizeWorkspace(null));
  const [expandedFolders, setExpandedFolders] = useState(() => new Set(["src"]));
  const [shareUrl, setShareUrl] = useState("");
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [isQrOpen, setIsQrOpen] = useState(false);
  const [presence, setPresence] = useState([]);
  const [connectionState, setConnectionState] = useState("disconnected");
  const [jobSummary, setJobSummary] = useState({
    text: "No job running",
    isError: false,
  });
  const [jobLogs, setJobLogs] = useState("Pipeline logs will appear here.");
  const [activeJob, setActiveJob] = useState(null);
  const [assets, setAssets] = useState([]);
  const [runGroups, setRunGroups] = useState([]);
  const [runActionBusy, setRunActionBusy] = useState("");
  const [launchingKind, setLaunchingKind] = useState("");
  const [modelCapabilities, setModelCapabilities] = useState(null);
  const [execOutput, setExecOutput] = useState({ run_id: '', stdout: '', stderr: '', exit_code: null, time_ms: 0, status: 'idle' });

  const wsRef = useRef(null);
  const editDebounceRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const pollTimerRef = useRef(null);
  const polledJobIdRef = useRef("");
  const activeProjectRef = useRef("");
  const workspaceRef = useRef(normalizeWorkspace(null));

  const setJobMessage = useCallback((text, isError = false) => {
    setJobSummary({ text, isError });
  }, []);

  const stopJobPolling = useCallback(() => {
    if (pollTimerRef.current) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    polledJobIdRef.current = "";
  }, []);

  useEffect(() => {
    workspaceRef.current = workspace;
  }, [workspace]);

  const renderJob = useCallback(
    (job) => {
      setActiveJob(job);
      const hasError = job.status === "failed";
      const kind = String(job.kind || "pipeline").replaceAll("_", " ");
      const renderNote = job?.result?.render_note ? ` | ${job.result.render_note}` : "";
      setJobMessage(
        `${kind.toUpperCase()} | ${String(job.status).toUpperCase()} (${job.progress}%)${renderNote}`,
        hasError
      );

      const logs = Array.isArray(job.logs) ? job.logs.slice(-40) : [];
      const errorLine = job.error ? `\nERROR: ${job.error}` : "";
      setJobLogs(`${logs.join("\n")}${errorLine}`.trim() || "No logs available.");
    },
    [setJobMessage]
  );

  const refreshAssets = useCallback(
    async (pid) => {
      if (!pid) {
        return;
      }

      try {
        const [assetRecords, runRecords] = await Promise.all([
          api(`/api/projects/${pid}/assets`),
          api(`/api/projects/${pid}/runs`).catch(() => null),
        ]);
        setAssets(Array.isArray(assetRecords) ? assetRecords : []);
        setRunGroups(Array.isArray(runRecords) ? runRecords : []);
      } catch (error) {
        setJobMessage(`Could not load assets: ${error.message}`, true);
      }
    },
    [setJobMessage]
  );

  const refreshModelCapabilities = useCallback(async () => {
    try {
      const capabilities = await api("/api/model-capabilities");
      setModelCapabilities(capabilities);
    } catch (error) {
      setJobMessage(`Could not load model capabilities: ${error.message}`, true);
    }
  }, [setJobMessage]);

  const applyWorkspaceSnapshot = useCallback((workspacePayload) => {
    const normalized = normalizeWorkspace(workspacePayload);
    setWorkspace(normalized);
    setExpandedFolders((previous) => {
      const next = new Set(previous);
      const activePath = normalized.active_file || "";
      const segments = activePath.split("/");
      let prefix = "";
      for (let index = 0; index < segments.length - 1; index += 1) {
        prefix = prefix ? `${prefix}/${segments[index]}` : segments[index];
        next.add(prefix);
      }
      return next;
    });
  }, []);

  const refreshWorkspace = useCallback(
    async (pid) => {
      if (!pid) {
        return;
      }

      try {
        const payload = await api(`/api/projects/${pid}/workspace`);
        applyWorkspaceSnapshot(payload);
      } catch (error) {
        setJobMessage(`Could not load workspace: ${error.message}`, true);
      }
    },
    [applyWorkspaceSnapshot, setJobMessage]
  );

  const expandFoldersForPath = useCallback((filePath) => {
    if (!filePath) {
      return;
    }
    setExpandedFolders((previous) => {
      const next = new Set(previous);
      const segments = String(filePath).split("/");
      let prefix = "";
      for (let index = 0; index < segments.length - 1; index += 1) {
        prefix = prefix ? `${prefix}/${segments[index]}` : segments[index];
        if (prefix) {
          next.add(prefix);
        }
      }
      return next;
    });
  }, []);

  const applyWorkspaceEvent = useCallback(
    (payload) => {
      const type = String(payload?.type || "");
      const timestamp = String(payload?.updated_at || new Date().toISOString());

      setWorkspace((previous) => {
        const base = normalizeWorkspace(previous);
        const files = { ...base.files };
        let activeFile = base.active_file;

        if (type === "file_edit") {
          const path = normalizeWorkspacePath(payload?.path || "");
          if (!path) {
            return base;
          }
          const existing = files[path] || {
            content: "",
            updated_at: timestamp,
            language: extensionToLanguage(path),
          };
          files[path] = {
            ...existing,
            content: String(payload?.content || ""),
            updated_at: timestamp,
            language: extensionToLanguage(path),
          };
          activeFile = normalizeWorkspacePath(payload?.active_file || path) || path;
        } else if (type === "file_create") {
          const path = normalizeWorkspacePath(payload?.path || "");
          if (!path) {
            return base;
          }
          files[path] = {
            content: String(payload?.content || ""),
            updated_at: timestamp,
            language: extensionToLanguage(path),
          };
          activeFile = normalizeWorkspacePath(payload?.active_file || path) || path;
        } else if (type === "file_delete") {
          const path = normalizeWorkspacePath(payload?.path || "");
          if (path && files[path]) {
            delete files[path];
          }
          if (!Object.keys(files).length) {
            files[DEFAULT_ACTIVE_FILE] = {
              content: DEFAULT_CODE,
              updated_at: timestamp,
              language: extensionToLanguage(DEFAULT_ACTIVE_FILE),
            };
            activeFile = DEFAULT_ACTIVE_FILE;
          } else {
            const hinted = normalizeWorkspacePath(payload?.active_file || "");
            activeFile = hinted && files[hinted] ? hinted : Object.keys(files).sort()[0];
          }
        } else if (type === "file_rename") {
          const oldPath = normalizeWorkspacePath(payload?.old_path || "");
          const newPath = normalizeWorkspacePath(payload?.new_path || "");
          if (!oldPath || !newPath || !files[oldPath]) {
            return base;
          }
          files[newPath] = {
            ...(files[oldPath] || { content: "" }),
            updated_at: timestamp,
            language: extensionToLanguage(newPath),
          };
          delete files[oldPath];
          const hinted = normalizeWorkspacePath(payload?.active_file || "");
          activeFile = hinted && files[hinted] ? hinted : newPath;
        } else if (type === "active_file") {
          const hinted = normalizeWorkspacePath(payload?.path || "");
          if (!hinted || !files[hinted]) {
            return base;
          }
          activeFile = hinted;
        } else {
          return base;
        }

        return normalizeWorkspace({
          files,
          active_file: activeFile,
          updated_at: timestamp,
        });
      });

      if (type === "file_edit" || type === "file_create" || type === "file_rename" || type === "active_file") {
        const hintedPath =
          normalizeWorkspacePath(payload?.active_file || "") ||
          normalizeWorkspacePath(payload?.path || "") ||
          normalizeWorkspacePath(payload?.new_path || "");
        expandFoldersForPath(hintedPath);
      }
    },
    [expandFoldersForPath]
  );

  const processIncomingJob = useCallback(
    async (job, pid) => {
      if (!job || typeof job !== "object") {
        return;
      }

      renderJob(job);
      const status = String(job.status || "");
      if (status === "completed" || status === "failed") {
        setLaunchingKind("");
        stopJobPolling();
        await refreshAssets(pid);
      }
    },
    [refreshAssets, renderJob, stopJobPolling]
  );

  const startJobPolling = useCallback(
    (jobId, pid) => {
      if (!jobId || !pid) {
        return;
      }
      if (polledJobIdRef.current === jobId && pollTimerRef.current) {
        return;
      }

      stopJobPolling();
      polledJobIdRef.current = jobId;

      pollTimerRef.current = window.setInterval(async () => {
        try {
          const job = await api(`/api/jobs/${jobId}`);
          await processIncomingJob(job, pid);
        } catch (error) {
          stopJobPolling();
          setJobMessage(`Job polling stopped: ${error.message}`, true);
        }
      }, 1200);
    },
    [processIncomingJob, setJobMessage, stopJobPolling]
  );

  const refreshJobs = useCallback(
    async (pid) => {
      if (!pid) {
        return;
      }

      try {
        const records = await api(`/api/projects/${pid}/jobs`);
        if (!Array.isArray(records) || !records.length) {
          return;
        }

        const latest = records[0];
        await processIncomingJob(latest, pid);
        if (latest.status === "running" || latest.status === "queued") {
          startJobPolling(latest.id, pid);
        }
      } catch (error) {
        setJobMessage(`Could not load jobs: ${error.message}`, true);
      }
    },
    [processIncomingJob, setJobMessage, startJobPolling]
  );

  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      const params = new URLSearchParams(window.location.search);
      const requestedProjectId = params.get("project");
      let resolvedProjectId = "";

      try {
        if (requestedProjectId) {
          const project = await api(`/api/projects/${requestedProjectId}`);
          resolvedProjectId = project.id;
          if (!cancelled) {
            applyWorkspaceSnapshot(project.workspace || null);
          }
        } else {
          const project = await api("/api/projects", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: "Slingshot Demo Project",
              initial_script: DEFAULT_SCRIPT,
            }),
          });
          resolvedProjectId = project.id;

          if (!cancelled) {
            applyWorkspaceSnapshot(project.workspace || null);
          }

          const url = new URL(window.location.href);
          url.searchParams.set("project", project.id);
          window.history.replaceState({}, "", url.toString());
        }
      } catch (error) {
        if (requestedProjectId) {
          setJobMessage(
            `Project ${requestedProjectId} not found. Creating a new project.`,
            true
          );

          const project = await api("/api/projects", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: "Slingshot Demo Project",
              initial_script: DEFAULT_SCRIPT,
            }),
          });
          resolvedProjectId = project.id;

          if (!cancelled) {
            applyWorkspaceSnapshot(project.workspace || null);
          }

          const url = new URL(window.location.href);
          url.searchParams.set("project", project.id);
          window.history.replaceState({}, "", url.toString());
        } else {
          setJobMessage(`Initialization failed: ${error.message}`, true);
          return;
        }
      }

      if (cancelled) {
        return;
      }

      setProjectId(resolvedProjectId);
      activeProjectRef.current = resolvedProjectId;
      await refreshWorkspace(resolvedProjectId);
      await refreshModelCapabilities();
      await refreshAssets(resolvedProjectId);
      await refreshJobs(resolvedProjectId);
    }

    initialize().catch((error) => {
      setJobMessage(`Initialization failed: ${error.message}`, true);
    });

    return () => {
      cancelled = true;
      if (editDebounceRef.current) {
        window.clearTimeout(editDebounceRef.current);
      }
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      stopJobPolling();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [
    applyWorkspaceSnapshot,
    refreshAssets,
    refreshJobs,
    refreshModelCapabilities,
    refreshWorkspace,
    setJobMessage,
    stopJobPolling,
  ]);

  useEffect(() => {
    if (!projectId) {
      return;
    }

    let disposed = false;

    const connect = () => {
      if (disposed) {
        return;
      }

      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/${projectId}/${clientId}`);
      wsRef.current = ws;
      setConnectionState("connecting");

      ws.addEventListener("open", () => {
        if (!disposed) {
          setConnectionState("connected");
        }
      });

      ws.addEventListener("message", (event) => {
        if (disposed) {
          return;
        }

        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          return;
        }

        if (payload.type === "state") {
          return;
        }

        if (payload.type === "workspace_state") {
          applyWorkspaceSnapshot(payload.workspace || null);
          return;
        }

        if (
          payload.type === "file_edit" ||
          payload.type === "file_create" ||
          payload.type === "file_delete" ||
          payload.type === "file_rename" ||
          payload.type === "active_file"
        ) {
          applyWorkspaceEvent(payload);
          if (payload.editor && payload.editor !== clientId) {
            setJobMessage(`Live workspace update from ${payload.editor}`);
          }
          return;
        }

        if (payload.type === "presence") {
          const clients = Array.isArray(payload.clients) ? payload.clients : [];
          setPresence(clients);
          return;
        }

        if (payload.type === "job_update") {
          const pid = String(payload.project_id || "");
          const job = payload.job;
          if (!pid || pid !== activeProjectRef.current || !job) {
            return;
          }

          const status = String(job.status || "");
          if (status === "queued" || status === "running") {
            startJobPolling(String(job.id || ""), pid);
          }

          processIncomingJob(job, pid).catch(() => {
            setJobMessage("Failed to process incoming job update.", true);
          });
          return;
        }

        if (payload.type === "error") {
          setJobMessage(payload.message || "Socket error received", true);
        }
      });

      ws.addEventListener("close", () => {
        if (disposed) {
          return;
        }

        setConnectionState("disconnected");
        reconnectTimerRef.current = window.setTimeout(() => {
          if (!disposed) {
            connect();
          }
        }, 1400);
      });

      ws.addEventListener("error", () => {
        if (!disposed) {
          setConnectionState("error");
        }
      });
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [
    applyWorkspaceEvent,
    applyWorkspaceSnapshot,
    clientId,
    processIncomingJob,
    projectId,
    setJobMessage,
    startJobPolling,
  ]);

  const sendWorkspaceMessage = useCallback((payload) => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN || !activeProjectRef.current) {
      return false;
    }
    socket.send(JSON.stringify(payload));
    return true;
  }, []);

  const selectFile = useCallback(
    (path) => {
      const normalized = normalizeWorkspacePath(path);
      if (!normalized) {
        return;
      }

      setWorkspace((previous) => {
        const base = normalizeWorkspace(previous);
        if (!base.files[normalized]) {
          return base;
        }
        return {
          ...base,
          active_file: normalized,
          updated_at: new Date().toISOString(),
        };
      });

      expandFoldersForPath(normalized);
      sendWorkspaceMessage({ type: "set_active_file", path: normalized });
    },
    [expandFoldersForPath, sendWorkspaceMessage]
  );

  const onCodeChange = useCallback(
    (event) => {
      const next = event.target.value;
      const currentWorkspace = workspaceRef.current;
      const activeFile = normalizeWorkspacePath(currentWorkspace?.active_file || "");
      if (!activeFile) {
        return;
      }

      setWorkspace((previous) => {
        const base = normalizeWorkspace(previous);
        const files = { ...base.files };
        const existing = files[activeFile] || {
          content: "",
          updated_at: new Date().toISOString(),
          language: extensionToLanguage(activeFile),
        };
        files[activeFile] = {
          ...existing,
          content: next,
          updated_at: new Date().toISOString(),
          language: extensionToLanguage(activeFile),
        };
        return {
          ...base,
          files,
          active_file: activeFile,
          updated_at: new Date().toISOString(),
        };
      });

      if (editDebounceRef.current) {
        window.clearTimeout(editDebounceRef.current);
      }

      editDebounceRef.current = window.setTimeout(() => {
        sendWorkspaceMessage({
          type: "file_edit",
          path: activeFile,
          content: next,
        });
      }, 90);
    },
    [sendWorkspaceMessage]
  );

  const createFile = useCallback(() => {
    const value = window.prompt("New file path (example: src/utils/helpers.py)", "src/new_file.py");
    if (!value) {
      return;
    }

    const normalized = normalizeWorkspacePath(value);
    if (!normalized) {
      setJobMessage("Invalid file path.", true);
      return;
    }

    if (workspaceRef.current.files && workspaceRef.current.files[normalized]) {
      setJobMessage("File already exists.", true);
      return;
    }

    const starter = extensionToLanguage(normalized) === "python" ? "" : "";
    setWorkspace((previous) => {
      const base = normalizeWorkspace(previous);
      const files = { ...base.files };
      files[normalized] = {
        content: starter,
        updated_at: new Date().toISOString(),
        language: extensionToLanguage(normalized),
      };
      return {
        ...base,
        files,
        active_file: normalized,
        updated_at: new Date().toISOString(),
      };
    });

    expandFoldersForPath(normalized);
    sendWorkspaceMessage({ type: "file_create", path: normalized, content: starter });
  }, [expandFoldersForPath, sendWorkspaceMessage, setJobMessage]);

  const renameActiveFile = useCallback(() => {
    const activeFile = normalizeWorkspacePath(workspaceRef.current?.active_file || "");
    if (!activeFile) {
      return;
    }

    const value = window.prompt("Rename file path", activeFile);
    if (!value) {
      return;
    }
    const nextPath = normalizeWorkspacePath(value);
    if (!nextPath) {
      setJobMessage("Invalid file path.", true);
      return;
    }
    if (nextPath === activeFile) {
      return;
    }
    if (workspaceRef.current.files && workspaceRef.current.files[nextPath]) {
      setJobMessage("A file already exists at that path.", true);
      return;
    }

    setWorkspace((previous) => {
      const base = normalizeWorkspace(previous);
      const files = { ...base.files };
      const entry = files[activeFile];
      if (!entry) {
        return base;
      }
      files[nextPath] = {
        ...entry,
        language: extensionToLanguage(nextPath),
        updated_at: new Date().toISOString(),
      };
      delete files[activeFile];
      return {
        ...base,
        files,
        active_file: nextPath,
        updated_at: new Date().toISOString(),
      };
    });

    expandFoldersForPath(nextPath);
    sendWorkspaceMessage({ type: "file_rename", old_path: activeFile, new_path: nextPath });
  }, [expandFoldersForPath, sendWorkspaceMessage, setJobMessage]);

  const deleteActiveFile = useCallback(() => {
    const activeFile = normalizeWorkspacePath(workspaceRef.current?.active_file || "");
    if (!activeFile) {
      return;
    }

    const confirmed = window.confirm(`Delete file '${activeFile}'?`);
    if (!confirmed) {
      return;
    }

    setWorkspace((previous) => {
      const base = normalizeWorkspace(previous);
      const files = { ...base.files };
      delete files[activeFile];

      if (!Object.keys(files).length) {
        files[DEFAULT_ACTIVE_FILE] = {
          content: DEFAULT_CODE,
          updated_at: new Date().toISOString(),
          language: extensionToLanguage(DEFAULT_ACTIVE_FILE),
        };
      }

      const nextActive = files[activeFile]
        ? activeFile
        : Object.keys(files).sort()[0] || DEFAULT_ACTIVE_FILE;

      return {
        ...base,
        files,
        active_file: nextActive,
        updated_at: new Date().toISOString(),
      };
    });

    sendWorkspaceMessage({ type: "file_delete", path: activeFile });
  }, [sendWorkspaceMessage]);

  const launchJob = useCallback(async (kind) => {
    if (!projectId) {
      return;
    }

    setLaunchingKind(kind);
    try {
      const job = await api(`/api/projects/${projectId}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind }),
      });
      renderJob(job);
      startJobPolling(job.id, projectId);
    } catch (error) {
      setLaunchingKind("");
      setJobMessage(`Failed to start ${kind.replaceAll("_", " ")} job: ${error.message}`, true);
      await refreshJobs(projectId);
    }
  }, [projectId, refreshJobs, renderJob, setJobMessage, startJobPolling]);

  const resolveShareUrl = useCallback(async () => {
    if (!projectId) {
      return "";
    }

    const url = new URL(window.location.href);
    url.searchParams.set("project", projectId);

    if (["localhost", "127.0.0.1", "0.0.0.0"].includes(url.hostname)) {
      try {
        const network = await api("/api/network");
        if (network?.lan_ip) {
          url.hostname = String(network.lan_ip);
        }
      } catch {
        // Keep localhost URL when network API is unavailable.
      }
    }

    return url.toString();
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;

    async function updateShareUrl() {
      if (!projectId) {
        setShareUrl("");
        return;
      }

      const url = await resolveShareUrl();
      if (!cancelled) {
        setShareUrl(url);
      }
    }

    updateShareUrl().catch(() => {
      if (!cancelled) {
        setShareUrl("");
      }
    });

    return () => {
      cancelled = true;
    };
  }, [projectId, resolveShareUrl]);

  useEffect(() => {
    let cancelled = false;

    async function buildQrCode() {
      if (!shareUrl) {
        setQrDataUrl("");
        return;
      }

      const dataUrl = await QRCode.toDataURL(shareUrl, {
        width: 260,
        margin: 1,
        errorCorrectionLevel: "M",
      });
      if (!cancelled) {
        setQrDataUrl(dataUrl);
      }
    }

    buildQrCode().catch(() => {
      if (!cancelled) {
        setQrDataUrl("");
      }
    });

    return () => {
      cancelled = true;
    };
  }, [shareUrl]);

  const copyProjectLink = useCallback(async () => {
    if (!projectId) {
      return;
    }

    const url = (await resolveShareUrl()) || window.location.href;

    try {
      await navigator.clipboard.writeText(url);
      setJobMessage("Project link copied. Share it to join the same room.");
    } catch {
      setJobMessage("Unable to access clipboard. Copy URL from browser manually.", true);
    }
  }, [projectId, resolveShareUrl, setJobMessage]);

  const openQrCode = useCallback(async () => {
    if (!projectId) {
      return;
    }

    if (!shareUrl) {
      const fallback = await resolveShareUrl();
      if (fallback) {
        setShareUrl(fallback);
      }
    }
    setIsQrOpen(true);
  }, [projectId, resolveShareUrl, shareUrl]);

  const downloadRunArchive = useCallback(
    (folder) => {
      if (!projectId || !folder) {
        return;
      }
      const url = `/api/projects/${projectId}/runs/${encodeURIComponent(folder)}/download`;
      window.open(url, "_blank", "noopener,noreferrer");
    },
    [projectId]
  );

  const deleteRunFolder = useCallback(
    async (folder) => {
      if (!projectId || !folder || folder === "ungrouped") {
        return;
      }

      const confirmed = window.confirm(`Delete run folder '${folder}' and all assets inside it?`);
      if (!confirmed) {
        return;
      }

      const busyKey = `delete:${folder}`;
      setRunActionBusy(busyKey);
      try {
        const result = await api(`/api/projects/${projectId}/runs/${encodeURIComponent(folder)}`, {
          method: "DELETE",
        });
        setJobMessage(
          `Deleted ${folder} (${result.removed_assets || 0} asset(s) removed).`
        );
        await refreshAssets(projectId);
      } catch (error) {
        setJobMessage(`Could not delete run folder: ${error.message}`, true);
      } finally {
        setRunActionBusy("");
      }
    },
    [projectId, refreshAssets, setJobMessage]
  );

  const cleanupOldRuns = useCallback(
    async (keepLatest = 3) => {
      if (!projectId) {
        return;
      }

      const confirmed = window.confirm(
        `Cleanup old runs and keep only latest ${keepLatest} run(s)?`
      );
      if (!confirmed) {
        return;
      }

      setRunActionBusy("cleanup");
      try {
        const result = await api(`/api/projects/${projectId}/runs/cleanup`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keep_latest: keepLatest, include_ungrouped: false }),
        });
        const skippedCount = Array.isArray(result.skipped_runs) ? result.skipped_runs.length : 0;
        setJobMessage(
          `Cleanup completed: removed ${result.removed_run_count || 0} run(s), ${result.removed_asset_count || 0} asset(s), skipped ${skippedCount} active run(s).`
        );
        await refreshAssets(projectId);
      } catch (error) {
        setJobMessage(`Could not cleanup runs: ${error.message}`, true);
      } finally {
        setRunActionBusy("");
      }
    },
    [projectId, refreshAssets, setJobMessage]
  );

  const collaboratorsLabel =
    presence.length > 0
      ? `Collaborators: ${presence.length} (${presence.join(", ")})`
      : "Collaborators: 0 (none)";

  const workspaceFiles = workspace && typeof workspace.files === "object" ? workspace.files : {};
  const workspacePaths = Object.keys(workspaceFiles).sort();
  const activeFilePath =
    normalizeWorkspacePath(workspace?.active_file || "") || workspacePaths[0] || DEFAULT_ACTIVE_FILE;
  const activeFileEntry = workspaceFiles[activeFilePath] || {
    content: "",
    language: extensionToLanguage(activeFilePath),
    updated_at: "",
  };
  const explorerTree = useMemo(() => buildExplorerTree(workspaceFiles), [workspaceFiles]);

  const toggleFolder = useCallback((folderPath) => {
    if (!folderPath) {
      return;
    }
    setExpandedFolders((previous) => {
      const next = new Set(previous);
      if (next.has(folderPath)) {
        next.delete(folderPath);
      } else {
        next.add(folderPath);
      }
      return next;
    });
  }, []);

  const renderExplorerNodes = useCallback(
    (nodes, depth = 0) =>
      nodes.map((node) => {
        if (node.type === "folder") {
          const isExpanded = expandedFolders.has(node.path);
          return (
            <div key={node.path || "root"}>
              {node.path ? (
                <button
                  type="button"
                  className={`explorer-node explorer-folder ${isExpanded ? "expanded" : ""}`}
                  style={{ paddingLeft: `${12 + depth * 14}px` }}
                  onClick={() => toggleFolder(node.path)}
                >
                  <span className="explorer-icon">{isExpanded ? "▾" : "▸"}</span>
                  <span>{node.name}</span>
                </button>
              ) : null}
              {(node.path === "" || isExpanded) && node.children?.length
                ? renderExplorerNodes(node.children, node.path === "" ? depth : depth + 1)
                : null}
            </div>
          );
        }

        const isActive = node.path === activeFilePath;
        return (
          <button
            key={node.path}
            type="button"
            className={`explorer-node explorer-file ${isActive ? "active" : ""}`}
            style={{ paddingLeft: `${12 + depth * 14}px` }}
            onClick={() => selectFile(node.path)}
          >
            <span className="explorer-icon">•</span>
            <span>{node.name}</span>
          </button>
        );
      }),
    [activeFilePath, expandedFolders, selectFile, toggleFolder]
  );

  const latestResult = activeJob && typeof activeJob.result === "object" ? activeJob.result : {};
  const configuredStack =
    modelCapabilities && modelCapabilities.configured && typeof modelCapabilities.configured === "object"
      ? modelCapabilities.configured
      : {};
  const systemMetrics =
    latestResult && latestResult.system_metrics && typeof latestResult.system_metrics === "object"
      ? latestResult.system_metrics
      : {};
  const runtimeProviders =
    latestResult && latestResult.model_runtime && typeof latestResult.model_runtime === "object"
      ? latestResult.model_runtime
      : {};
  const stageTimingEntries =
    latestResult && latestResult.timings_s && typeof latestResult.timings_s === "object"
      ? Object.entries(latestResult.timings_s)
      : [];
  const trainingTrace = Array.isArray(latestResult.training_trace)
    ? latestResult.training_trace
    : [];
  const configuredSummary = Object.entries(configuredStack)
    .map(([name, provider]) => `${name}: ${provider}`)
    .join(" | ");
  const runtimeProviderEntries = Object.entries(runtimeProviders);
  const assetGroups = useMemo(() => {
    if (Array.isArray(runGroups) && runGroups.length) {
      return runGroups;
    }
    return groupAssetsByFolder(assets);
  }, [assets, runGroups]);

  return (
    <>
      <div className="noise" />
      <main className="layout">
        <section className="panel hero">
          <p className="eyebrow">Distributed Multimodal Collaboration</p>
          <h1>Slingshot Prototype Runner</h1>
          <p className="subtitle">
            Live collaborative coding workspace with local GPU pipeline and shared run outputs.
          </p>
          <div className="meta-row">
            <span className="badge">Project: {projectId || "initializing..."}</span>
            <span className="badge">Client: {clientId}</span>
            <span className="badge">Socket: {connectionState}</span>
          </div>
          <div className="hero-actions">
            <button type="button" onClick={copyProjectLink} disabled={!projectId}>
              Copy project link
            </button>
            <button
              type="button"
              className="secondary-btn"
              onClick={openQrCode}
              disabled={!projectId}
            >
              Show QR code
            </button>
          </div>
          <p className="hint share-url">Share URL: {shareUrl || "preparing..."}</p>
        </section>

        <section className="panel editor-panel">
          <div className="panel-head">
            <h2>Collaborative Coding</h2>
            <span className="presence-label">{collaboratorsLabel}</span>
          </div>

          <div className="editor-actions">
            <button type="button" onClick={createFile} disabled={!projectId}>
              New file
            </button>
            <button type="button" className="secondary-btn" onClick={renameActiveFile} disabled={!projectId}>
              Rename file
            </button>
            <button type="button" className="danger-btn" onClick={deleteActiveFile} disabled={!projectId}>
              Delete file
            </button>
          </div>

          <div className="workspace-shell">
            <aside className="explorer-pane">
              <div className="explorer-header">Explorer</div>
              <div className="explorer-tree">{renderExplorerNodes(explorerTree.children || [])}</div>
            </aside>

            <div className="code-pane">
              <div className="code-tabs">
                {workspacePaths.map((path) => (
                  <button
                    type="button"
                    key={path}
                    className={`code-tab ${path === activeFilePath ? "active" : ""}`}
                    onClick={() => selectFile(path)}
                  >
                    {path.split("/").pop()}
                  </button>
                ))}
              </div>

              <div className="code-meta">
                <span>{activeFilePath}</span>
                <span>Language: {activeFileEntry.language || extensionToLanguage(activeFilePath)}</span>
              </div>

              <textarea
                id="codeEditor"
                spellCheck="false"
                value={String(activeFileEntry.content || "")}
                onChange={onCodeChange}
                placeholder="Start coding here..."
              />
            </div>
          </div>

          <p className="hint">Open this project link on another device to co-edit the same code in real time.</p>
        </section>

        <section className="panel pipeline-panel">
          <div className="panel-head">
            <h2>Pipeline Job</h2>
            <div className="job-actions">
              <button
                type="button"
                onClick={() => launchJob("pipeline")}
                disabled={!projectId || Boolean(launchingKind)}
              >
                {launchingKind === "pipeline" ? "Starting..." : "Run pipeline"}
              </button>
              <button
                type="button"
                className="secondary-btn"
                onClick={() => launchJob("fine_tune")}
                disabled={!projectId || Boolean(launchingKind)}
              >
                {launchingKind === "fine_tune" ? "Starting..." : "Run fine-tune"}
              </button>
            </div>
          </div>
          <div className={`job-pill ${jobSummary.isError ? "job-pill-error" : ""}`}>
            {jobSummary.text}
          </div>
          {activeJob ? (
            <div className="job-details">
              <p className="asset-meta">
                Job ID: {activeJob.id} | Kind: {String(activeJob.kind || "pipeline").replaceAll("_", " ")}
              </p>
              {typeof latestResult.total_duration_s === "number" ? (
                <p className="asset-meta">Total runtime: {latestResult.total_duration_s}s</p>
              ) : null}
              {systemMetrics.gpu_backend ? (
                <p className="asset-meta">
                  Acceleration: {String(systemMetrics.gpu_backend)}
                  {systemMetrics.gpu_tool ? ` (${systemMetrics.gpu_tool})` : ""}
                </p>
              ) : null}
              {systemMetrics.gpu_summary ? (
                <p className="asset-meta">GPU: {String(systemMetrics.gpu_summary)}</p>
              ) : null}
              {runtimeProviderEntries.length ? (
                <div className="timings-grid">
                  {runtimeProviderEntries.map(([stage, details]) => {
                    const payload = details && typeof details === "object" ? details : {};
                    const provider = payload.provider || "mock";
                    const mode = payload.mode || "synthetic";
                    return (
                      <span key={`provider-${stage}`} className="timing-chip">
                        {stage}: {provider} ({mode})
                      </span>
                    );
                  })}
                </div>
              ) : null}
              {stageTimingEntries.length ? (
                <div className="timings-grid">
                  {stageTimingEntries.map(([stage, seconds]) => (
                    <span key={stage} className="timing-chip">
                      {stage.replaceAll("_", " ")}: {seconds}s
                    </span>
                  ))}
                </div>
              ) : null}
              {trainingTrace.length ? (
                <div className="timings-grid">
                  {trainingTrace.map((entry) => (
                    <span key={`epoch-${entry.epoch}`} className="timing-chip">
                      E{entry.epoch}: train {entry.train_loss} | val {entry.validation_loss}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
          <pre id="jobLogs">{jobLogs}</pre>
        </section>

        <section className="panel assets-panel">
          <div className="panel-head">
            <h2>Generated Assets</h2>
            <div className="job-actions">
              <button
                type="button"
                onClick={() => refreshAssets(projectId)}
                disabled={!projectId || Boolean(runActionBusy)}
              >
                Refresh
              </button>
              <button
                type="button"
                className="secondary-btn"
                onClick={() => cleanupOldRuns(3)}
                disabled={!projectId || Boolean(runActionBusy)}
              >
                {runActionBusy === "cleanup" ? "Cleaning..." : "Cleanup old runs"}
              </button>
            </div>
          </div>

          {!assets.length ? (
            <p className="empty">No generated assets yet. Run the pipeline to create outputs.</p>
          ) : (
            <div className="asset-folder-list">
              {assetGroups.map((group) => (
                <section className="asset-folder" key={group.folder}>
                  <div className="asset-folder-head">
                    <h3 className="asset-folder-title">{group.folder}</h3>
                    <div className="folder-actions">
                      <span className="folder-count">{group.asset_count || group.assets.length} asset(s)</span>
                      {group.folder !== "ungrouped" ? (
                        <>
                          <button
                            type="button"
                            className="folder-download-btn"
                            onClick={() => downloadRunArchive(group.folder)}
                            disabled={Boolean(runActionBusy)}
                          >
                            Download zip
                          </button>
                          <button
                            type="button"
                            className="folder-delete-btn"
                            onClick={() => deleteRunFolder(group.folder)}
                            disabled={Boolean(runActionBusy)}
                          >
                            {runActionBusy === `delete:${group.folder}` ? "Deleting..." : "Delete run"}
                          </button>
                        </>
                      ) : null}
                    </div>
                  </div>
                  <p className="asset-meta">
                    Run type: {String(group.kind || "mixed").replaceAll("_", " ")}
                    {group.run_id ? ` | Run ID: ${group.run_id}` : ""}
                    {group.job && typeof group.job === "object"
                      ? ` | Job: ${String(group.job.status || "").toUpperCase()} (${group.job.progress || 0}%)`
                      : ""}
                  </p>

                  <div className="assets-grid">
                    {group.assets.map((asset) => (
                      <article className="asset-card" key={asset.asset_id}>
                        <h3>{asset.label || asset.filename || "Asset"}</h3>
                        <p className="asset-meta">
                          {asset.kind || "file"} | {asset.filename || "unknown"}
                        </p>

                        {asset.kind === "image" ? (
                          <img src={asset.url} alt={asset.label || "Generated image"} loading="lazy" />
                        ) : null}
                        {asset.kind === "audio" ? <audio src={asset.url} controls preload="none" /> : null}
                        {asset.kind === "video" ? <video src={asset.url} controls preload="metadata" /> : null}

                        <a className="asset-link" href={asset.url} target="_blank" rel="noreferrer">
                          Open asset
                        </a>
                      </article>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </section>
      </main>

      {isQrOpen ? (
        <div className="qr-overlay" onClick={() => setIsQrOpen(false)}>
          <section className="qr-modal" onClick={(event) => event.stopPropagation()}>
            <h3>Scan to Open on Mobile</h3>
            {qrDataUrl ? (
              <img className="qr-image" src={qrDataUrl} alt="Project share QR code" />
            ) : (
              <p className="hint">Generating QR code...</p>
            )}
            <p className="hint share-url">{shareUrl || "Share URL unavailable"}</p>
            <button type="button" className="secondary-btn" onClick={() => setIsQrOpen(false)}>
              Close
            </button>
          </section>
        </div>
      ) : null}
    </>
  );
}
