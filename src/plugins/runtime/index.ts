import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createJiti } from "jiti";
import { resolveEffectiveMessagesConfig, resolveHumanDelayConfig } from "../../agents/identity.js";
import { createMemoryGetTool, createMemorySearchTool } from "../../agents/tools/memory-tool.js";
import {
  chunkByNewline,
  chunkMarkdownText,
  chunkMarkdownTextWithMode,
  chunkText,
  chunkTextWithMode,
  resolveChunkMode,
  resolveTextChunkLimit,
} from "../../auto-reply/chunk.js";
import {
  hasControlCommand,
  isControlCommandMessage,
  shouldComputeCommandAuthorized,
} from "../../auto-reply/command-detection.js";
import { shouldHandleTextCommands } from "../../auto-reply/commands-registry.js";
import { withReplyDispatcher } from "../../auto-reply/dispatch.js";
import {
  formatAgentEnvelope,
  formatInboundEnvelope,
  resolveEnvelopeFormatOptions,
} from "../../auto-reply/envelope.js";
import {
  createInboundDebouncer,
  resolveInboundDebounceMs,
} from "../../auto-reply/inbound-debounce.js";
import { dispatchReplyFromConfig } from "../../auto-reply/reply/dispatch-from-config.js";
import { finalizeInboundContext } from "../../auto-reply/reply/inbound-context.js";
import {
  buildMentionRegexes,
  matchesMentionPatterns,
  matchesMentionWithExplicit,
} from "../../auto-reply/reply/mentions.js";
import { dispatchReplyWithBufferedBlockDispatcher } from "../../auto-reply/reply/provider-dispatcher.js";
import { createReplyDispatcherWithTyping } from "../../auto-reply/reply/reply-dispatcher.js";
import { removeAckReactionAfterReply, shouldAckReaction } from "../../channels/ack-reactions.js";
import { resolveCommandAuthorizedFromAuthorizers } from "../../channels/command-gating.js";
import { recordInboundSession } from "../../channels/session.js";
import { registerMemoryCli } from "../../cli/memory-cli.js";
import { loadConfig, writeConfigFile } from "../../config/config.js";
import {
  resolveChannelGroupPolicy,
  resolveChannelGroupRequireMention,
} from "../../config/group-policy.js";
import { resolveMarkdownTableMode } from "../../config/markdown-tables.js";
import { resolveStateDir } from "../../config/paths.js";
import {
  readSessionUpdatedAt,
  recordSessionMetaFromInbound,
  resolveStorePath,
  updateLastRoute,
} from "../../config/sessions.js";
import { shouldLogVerbose } from "../../globals.js";
import { getChannelActivity, recordChannelActivity } from "../../infra/channel-activity.js";
import { enqueueSystemEvent } from "../../infra/system-events.js";
import { getChildLogger } from "../../logging.js";
import { normalizeLogLevel } from "../../logging/levels.js";
import { convertMarkdownTables } from "../../markdown/tables.js";
import { isVoiceCompatibleAudio } from "../../media/audio.js";
import { mediaKindFromMime } from "../../media/constants.js";
import { fetchRemoteMedia } from "../../media/fetch.js";
import { getImageMetadata, resizeToJpeg } from "../../media/image-ops.js";
import { detectMime } from "../../media/mime.js";
import { saveMediaBuffer } from "../../media/store.js";
import { buildPairingReply } from "../../pairing/pairing-messages.js";
import {
  readChannelAllowFromStore,
  upsertChannelPairingRequest,
} from "../../pairing/pairing-store.js";
import { runCommandWithTimeout } from "../../process/exec.js";
import { resolveAgentRoute } from "../../routing/resolve-route.js";
import { textToSpeechTelephony } from "../../tts/tts.js";
import { getActiveWebListener } from "../../web/active-listener.js";
import {
  getWebAuthAgeMs,
  logoutWeb,
  logWebSelfId,
  readWebSelfId,
  webAuthExists,
} from "../../web/auth-store.js";
import { formatNativeDependencyHint } from "./native-deps.js";
import type { PluginRuntime } from "./types.js";

const runtimeRequire = createRequire(import.meta.url);
let runtimeJiti: ReturnType<typeof createJiti> | null = null;

let cachedVersion: string | null = null;

type RuntimeModuleRef = {
  srcFile: string;
  distFile: string;
  modulePath?: string;
};

const runtimeModulePathCache = new Map<string, string>();

function getRuntimeJiti(): ReturnType<typeof createJiti> {
  if (runtimeJiti) {
    return runtimeJiti;
  }
  runtimeJiti = createJiti(import.meta.url, {
    interopDefault: true,
    extensions: [".ts", ".tsx", ".mts", ".cts", ".js", ".mjs", ".cjs", ".json"],
  });
  return runtimeJiti;
}

function resolveRuntimeModuleFile(params: RuntimeModuleRef): string | null {
  const cacheKey = `${params.srcFile}::${params.distFile}`;
  const cached = runtimeModulePathCache.get(cacheKey);
  if (cached && fs.existsSync(cached)) {
    return cached;
  }
  try {
    const modulePath = params.modulePath ?? fileURLToPath(import.meta.url);
    const normalizedModulePath = modulePath.replace(/\\/g, "/");
    const isDistRuntime = normalizedModulePath.includes("/dist/");
    const isProduction = process.env.NODE_ENV === "production";
    const isTest = process.env.VITEST || process.env.NODE_ENV === "test";
    let cursor = path.dirname(modulePath);
    for (let i = 0; i < 8; i += 1) {
      const srcCandidate = path.join(cursor, params.srcFile);
      const distCandidate = path.join(cursor, params.distFile);
      const orderedCandidates = isDistRuntime
        ? [distCandidate, srcCandidate]
        : isProduction
          ? isTest
            ? [distCandidate, srcCandidate]
            : [distCandidate]
          : [srcCandidate, distCandidate];
      for (const candidate of orderedCandidates) {
        if (fs.existsSync(candidate)) {
          runtimeModulePathCache.set(cacheKey, candidate);
          return candidate;
        }
      }
      const parent = path.dirname(cursor);
      if (parent === cursor) {
        break;
      }
      cursor = parent;
    }
  } catch {
    // ignore
  }
  return null;
}

function requireRuntimeModule<T>(moduleRef: RuntimeModuleRef): T {
  const resolved = resolveRuntimeModuleFile(moduleRef);
  if (!resolved) {
    throw new Error(
      `Failed to resolve runtime module: ${moduleRef.distFile} (source fallback: ${moduleRef.srcFile})`,
    );
  }
  try {
    return runtimeRequire(resolved) as T;
  } catch (error) {
    // `createRequire` cannot always execute TypeScript source modules in dev/test; use Jiti there.
    if (resolved.endsWith(".ts") || resolved.endsWith(".mts") || resolved.endsWith(".cts")) {
      return getRuntimeJiti()(resolved) as T;
    }
    throw error;
  }
}

function createLazyAdapter<T extends object>(load: () => T): T {
  return new Proxy({} as T, {
    get(_target, prop, receiver) {
      return Reflect.get(load(), prop, receiver);
    },
  });
}

function resolveVersion(): string {
  if (cachedVersion) {
    return cachedVersion;
  }
  try {
    const pkg = runtimeRequire("../../../package.json") as { version?: string };
    cachedVersion = pkg.version ?? "unknown";
    return cachedVersion;
  } catch {
    cachedVersion = "unknown";
    return cachedVersion;
  }
}

const sendMessageWhatsAppLazy: PluginRuntime["channel"]["whatsapp"]["sendMessageWhatsApp"] = async (
  ...args
) => {
  const { sendMessageWhatsApp } = await loadWebOutbound();
  return sendMessageWhatsApp(...args);
};

const sendPollWhatsAppLazy: PluginRuntime["channel"]["whatsapp"]["sendPollWhatsApp"] = async (
  ...args
) => {
  const { sendPollWhatsApp } = await loadWebOutbound();
  return sendPollWhatsApp(...args);
};

const loginWebLazy: PluginRuntime["channel"]["whatsapp"]["loginWeb"] = async (...args) => {
  const { loginWeb } = await loadWebLogin();
  return loginWeb(...args);
};

const startWebLoginWithQrLazy: PluginRuntime["channel"]["whatsapp"]["startWebLoginWithQr"] = async (
  ...args
) => {
  const { startWebLoginWithQr } = await loadWebLoginQr();
  return startWebLoginWithQr(...args);
};

const waitForWebLoginLazy: PluginRuntime["channel"]["whatsapp"]["waitForWebLogin"] = async (
  ...args
) => {
  const { waitForWebLogin } = await loadWebLoginQr();
  return waitForWebLogin(...args);
};

const monitorWebChannelLazy: PluginRuntime["channel"]["whatsapp"]["monitorWebChannel"] = async (
  ...args
) => {
  const { monitorWebChannel } = await loadWebChannel();
  return monitorWebChannel(...args);
};

const handleWhatsAppActionLazy: PluginRuntime["channel"]["whatsapp"]["handleWhatsAppAction"] =
  async (...args) => {
    const { handleWhatsAppAction } = await loadWhatsAppActions();
    return handleWhatsAppAction(...args);
  };

const createWhatsAppLoginToolLazy: PluginRuntime["channel"]["whatsapp"]["createLoginTool"] = (
  ...args
) => {
  const { createWhatsAppLoginTool } = loadWhatsAppAgentToolsSync();
  return createWhatsAppLoginTool(...args);
};

const loadWebMediaLazy: PluginRuntime["media"]["loadWebMedia"] = async (...args) => {
  const { loadWebMedia } = await loadWebMediaModule();
  return loadWebMedia(...args);
};

const discordMessageActionsLazy = createLazyAdapter<
  PluginRuntime["channel"]["discord"]["messageActions"]
>(() => loadDiscordActionsSync().discordMessageActions);
const signalMessageActionsLazy = createLazyAdapter<
  PluginRuntime["channel"]["signal"]["messageActions"]
>(() => loadSignalActionsSync().signalMessageActions);
const telegramMessageActionsLazy = createLazyAdapter<
  PluginRuntime["channel"]["telegram"]["messageActions"]
>(() => loadTelegramActionsSync().telegramMessageActions);

const auditDiscordChannelPermissionsLazy: PluginRuntime["channel"]["discord"]["auditChannelPermissions"] =
  async (...args) => {
    const { auditDiscordChannelPermissions } = await loadDiscordAudit();
    return auditDiscordChannelPermissions(...args);
  };

const listDiscordDirectoryGroupsLiveLazy: PluginRuntime["channel"]["discord"]["listDirectoryGroupsLive"] =
  async (...args) => {
    const { listDiscordDirectoryGroupsLive } = await loadDiscordDirectoryLive();
    return listDiscordDirectoryGroupsLive(...args);
  };

const listDiscordDirectoryPeersLiveLazy: PluginRuntime["channel"]["discord"]["listDirectoryPeersLive"] =
  async (...args) => {
    const { listDiscordDirectoryPeersLive } = await loadDiscordDirectoryLive();
    return listDiscordDirectoryPeersLive(...args);
  };

const probeDiscordLazy: PluginRuntime["channel"]["discord"]["probeDiscord"] = async (...args) => {
  const { probeDiscord } = await loadDiscordProbe();
  return probeDiscord(...args);
};

const resolveDiscordChannelAllowlistLazy: PluginRuntime["channel"]["discord"]["resolveChannelAllowlist"] =
  async (...args) => {
    const { resolveDiscordChannelAllowlist } = await loadDiscordResolveChannels();
    return resolveDiscordChannelAllowlist(...args);
  };

const resolveDiscordUserAllowlistLazy: PluginRuntime["channel"]["discord"]["resolveUserAllowlist"] =
  async (...args) => {
    const { resolveDiscordUserAllowlist } = await loadDiscordResolveUsers();
    return resolveDiscordUserAllowlist(...args);
  };

const sendMessageDiscordLazy: PluginRuntime["channel"]["discord"]["sendMessageDiscord"] = async (
  ...args
) => {
  const { sendMessageDiscord } = await loadDiscordSend();
  return sendMessageDiscord(...args);
};

const sendPollDiscordLazy: PluginRuntime["channel"]["discord"]["sendPollDiscord"] = async (
  ...args
) => {
  const { sendPollDiscord } = await loadDiscordSend();
  return sendPollDiscord(...args);
};

const monitorDiscordProviderLazy: PluginRuntime["channel"]["discord"]["monitorDiscordProvider"] =
  async (...args) => {
    const { monitorDiscordProvider } = await loadDiscordMonitor();
    return monitorDiscordProvider(...args);
  };

const listSlackDirectoryGroupsLiveLazy: PluginRuntime["channel"]["slack"]["listDirectoryGroupsLive"] =
  async (...args) => {
    const { listSlackDirectoryGroupsLive } = await loadSlackDirectoryLive();
    return listSlackDirectoryGroupsLive(...args);
  };

const listSlackDirectoryPeersLiveLazy: PluginRuntime["channel"]["slack"]["listDirectoryPeersLive"] =
  async (...args) => {
    const { listSlackDirectoryPeersLive } = await loadSlackDirectoryLive();
    return listSlackDirectoryPeersLive(...args);
  };

const probeSlackLazy: PluginRuntime["channel"]["slack"]["probeSlack"] = async (...args) => {
  const { probeSlack } = await loadSlackProbe();
  return probeSlack(...args);
};

const resolveSlackChannelAllowlistLazy: PluginRuntime["channel"]["slack"]["resolveChannelAllowlist"] =
  async (...args) => {
    const { resolveSlackChannelAllowlist } = await loadSlackResolveChannels();
    return resolveSlackChannelAllowlist(...args);
  };

const resolveSlackUserAllowlistLazy: PluginRuntime["channel"]["slack"]["resolveUserAllowlist"] =
  async (...args) => {
    const { resolveSlackUserAllowlist } = await loadSlackResolveUsers();
    return resolveSlackUserAllowlist(...args);
  };

const sendMessageSlackLazy: PluginRuntime["channel"]["slack"]["sendMessageSlack"] = async (
  ...args
) => {
  const { sendMessageSlack } = await loadSlackSend();
  return sendMessageSlack(...args);
};

const monitorSlackProviderLazy: PluginRuntime["channel"]["slack"]["monitorSlackProvider"] = async (
  ...args
) => {
  const { monitorSlackProvider } = await loadSlackMonitor();
  return monitorSlackProvider(...args);
};

const handleSlackActionLazy: PluginRuntime["channel"]["slack"]["handleSlackAction"] = async (
  ...args
) => {
  const { handleSlackAction } = await loadSlackActions();
  return handleSlackAction(...args);
};

const collectTelegramUnmentionedGroupIdsLazy: PluginRuntime["channel"]["telegram"]["collectUnmentionedGroupIds"] =
  (...args) => {
    const { collectTelegramUnmentionedGroupIds } = loadTelegramAuditSync();
    return collectTelegramUnmentionedGroupIds(...args);
  };

const auditTelegramGroupMembershipLazy: PluginRuntime["channel"]["telegram"]["auditGroupMembership"] =
  async (...args) => {
    const { auditTelegramGroupMembership } = loadTelegramAuditSync();
    return auditTelegramGroupMembership(...args);
  };

const probeTelegramLazy: PluginRuntime["channel"]["telegram"]["probeTelegram"] = async (
  ...args
) => {
  const { probeTelegram } = await loadTelegramProbe();
  return probeTelegram(...args);
};

const resolveTelegramTokenLazy: PluginRuntime["channel"]["telegram"]["resolveTelegramToken"] = (
  ...args
) => {
  const { resolveTelegramToken } = loadTelegramTokenSync();
  return resolveTelegramToken(...args);
};

const sendMessageTelegramLazy: PluginRuntime["channel"]["telegram"]["sendMessageTelegram"] = async (
  ...args
) => {
  const { sendMessageTelegram } = await loadTelegramSend();
  return sendMessageTelegram(...args);
};

const sendPollTelegramLazy: PluginRuntime["channel"]["telegram"]["sendPollTelegram"] = async (
  ...args
) => {
  const { sendPollTelegram } = await loadTelegramSend();
  return sendPollTelegram(...args);
};

const monitorTelegramProviderLazy: PluginRuntime["channel"]["telegram"]["monitorTelegramProvider"] =
  async (...args) => {
    const { monitorTelegramProvider } = await loadTelegramMonitor();
    return monitorTelegramProvider(...args);
  };

const probeSignalLazy: PluginRuntime["channel"]["signal"]["probeSignal"] = async (...args) => {
  const { probeSignal } = await loadSignalProbe();
  return probeSignal(...args);
};

const sendMessageSignalLazy: PluginRuntime["channel"]["signal"]["sendMessageSignal"] = async (
  ...args
) => {
  const { sendMessageSignal } = await loadSignalSend();
  return sendMessageSignal(...args);
};

const monitorSignalProviderLazy: PluginRuntime["channel"]["signal"]["monitorSignalProvider"] =
  async (...args) => {
    const { monitorSignalProvider } = await loadSignalMonitor();
    return monitorSignalProvider(...args);
  };

const monitorIMessageProviderLazy: PluginRuntime["channel"]["imessage"]["monitorIMessageProvider"] =
  async (...args) => {
    const { monitorIMessageProvider } = await loadIMessageMonitor();
    return monitorIMessageProvider(...args);
  };

const probeIMessageLazy: PluginRuntime["channel"]["imessage"]["probeIMessage"] = async (
  ...args
) => {
  const { probeIMessage } = await loadIMessageProbe();
  return probeIMessage(...args);
};

const sendMessageIMessageLazy: PluginRuntime["channel"]["imessage"]["sendMessageIMessage"] = async (
  ...args
) => {
  const { sendMessageIMessage } = await loadIMessageSend();
  return sendMessageIMessage(...args);
};

const listLineAccountIdsLazy: PluginRuntime["channel"]["line"]["listLineAccountIds"] = (
  ...args
) => {
  const { listLineAccountIds } = loadLineAccountsSync();
  return listLineAccountIds(...args);
};

const resolveDefaultLineAccountIdLazy: PluginRuntime["channel"]["line"]["resolveDefaultLineAccountId"] =
  (...args) => {
    const { resolveDefaultLineAccountId } = loadLineAccountsSync();
    return resolveDefaultLineAccountId(...args);
  };

const resolveLineAccountLazy: PluginRuntime["channel"]["line"]["resolveLineAccount"] = (
  ...args
) => {
  const { resolveLineAccount } = loadLineAccountsSync();
  return resolveLineAccount(...args);
};

const normalizeLineAccountIdLazy: PluginRuntime["channel"]["line"]["normalizeAccountId"] = (
  ...args
) => {
  const { normalizeAccountId } = loadLineAccountsSync();
  return normalizeAccountId(...args);
};

const probeLineBotLazy: PluginRuntime["channel"]["line"]["probeLineBot"] = async (...args) => {
  const { probeLineBot } = await loadLineProbe();
  return probeLineBot(...args);
};

const sendMessageLineLazy: PluginRuntime["channel"]["line"]["sendMessageLine"] = async (
  ...args
) => {
  const { sendMessageLine } = loadLineSendSync();
  return sendMessageLine(...args);
};

const pushMessageLineLazy: PluginRuntime["channel"]["line"]["pushMessageLine"] = async (
  ...args
) => {
  const { pushMessageLine } = loadLineSendSync();
  return pushMessageLine(...args);
};

const pushMessagesLineLazy: PluginRuntime["channel"]["line"]["pushMessagesLine"] = async (
  ...args
) => {
  const { pushMessagesLine } = loadLineSendSync();
  return pushMessagesLine(...args);
};

const pushFlexMessageLazy: PluginRuntime["channel"]["line"]["pushFlexMessage"] = async (
  ...args
) => {
  const { pushFlexMessage } = loadLineSendSync();
  return pushFlexMessage(...args);
};

const pushTemplateMessageLazy: PluginRuntime["channel"]["line"]["pushTemplateMessage"] = async (
  ...args
) => {
  const { pushTemplateMessage } = loadLineSendSync();
  return pushTemplateMessage(...args);
};

const pushLocationMessageLazy: PluginRuntime["channel"]["line"]["pushLocationMessage"] = async (
  ...args
) => {
  const { pushLocationMessage } = loadLineSendSync();
  return pushLocationMessage(...args);
};

const pushTextMessageWithQuickRepliesLazy: PluginRuntime["channel"]["line"]["pushTextMessageWithQuickReplies"] =
  async (...args) => {
    const { pushTextMessageWithQuickReplies } = loadLineSendSync();
    return pushTextMessageWithQuickReplies(...args);
  };

const createQuickReplyItemsLazy: PluginRuntime["channel"]["line"]["createQuickReplyItems"] = (
  ...args
) => {
  const { createQuickReplyItems } = loadLineSendSync();
  return createQuickReplyItems(...args);
};

const buildTemplateMessageFromPayloadLazy: PluginRuntime["channel"]["line"]["buildTemplateMessageFromPayload"] =
  (...args) => {
    const { buildTemplateMessageFromPayload } = loadLineTemplateMessagesSync();
    return buildTemplateMessageFromPayload(...args);
  };

const monitorLineProviderLazy: PluginRuntime["channel"]["line"]["monitorLineProvider"] = async (
  ...args
) => {
  const { monitorLineProvider } = await loadLineMonitor();
  return monitorLineProvider(...args);
};

let webMediaPromise: Promise<typeof import("../../web/media.js")> | null = null;
let webOutboundPromise: Promise<typeof import("../../web/outbound.js")> | null = null;
let webLoginPromise: Promise<typeof import("../../web/login.js")> | null = null;
let webLoginQrPromise: Promise<typeof import("../../web/login-qr.js")> | null = null;
let webChannelPromise: Promise<typeof import("../../channels/web/index.js")> | null = null;
let whatsappActionsPromise: Promise<
  typeof import("../../agents/tools/whatsapp-actions.js")
> | null = null;

let whatsappAgentToolsModule:
  | typeof import("../../channels/plugins/agent-tools/whatsapp-login.js")
  | null = null;
let discordActionsModule: typeof import("../../channels/plugins/actions/discord.js") | null = null;
let signalActionsModule: typeof import("../../channels/plugins/actions/signal.js") | null = null;
let telegramActionsModule: typeof import("../../channels/plugins/actions/telegram.js") | null =
  null;
let telegramAuditModule: typeof import("../../telegram/audit.js") | null = null;
let telegramTokenModule: typeof import("../../telegram/token.js") | null = null;
let lineAccountsModule: typeof import("../../line/accounts.js") | null = null;
let lineSendModule: typeof import("../../line/send.js") | null = null;
let lineTemplateMessagesModule: typeof import("../../line/template-messages.js") | null = null;

let discordAuditPromise: Promise<typeof import("../../discord/audit.js")> | null = null;
let discordDirectoryLivePromise: Promise<typeof import("../../discord/directory-live.js")> | null =
  null;
let discordMonitorPromise: Promise<typeof import("../../discord/monitor.js")> | null = null;
let discordProbePromise: Promise<typeof import("../../discord/probe.js")> | null = null;
let discordResolveChannelsPromise: Promise<
  typeof import("../../discord/resolve-channels.js")
> | null = null;
let discordResolveUsersPromise: Promise<typeof import("../../discord/resolve-users.js")> | null =
  null;
let discordSendPromise: Promise<typeof import("../../discord/send.js")> | null = null;

let slackActionsPromise: Promise<typeof import("../../agents/tools/slack-actions.js")> | null =
  null;
let slackDirectoryLivePromise: Promise<typeof import("../../slack/directory-live.js")> | null =
  null;
let slackMonitorPromise: Promise<typeof import("../../slack/index.js")> | null = null;
let slackProbePromise: Promise<typeof import("../../slack/probe.js")> | null = null;
let slackResolveChannelsPromise: Promise<typeof import("../../slack/resolve-channels.js")> | null =
  null;
let slackResolveUsersPromise: Promise<typeof import("../../slack/resolve-users.js")> | null = null;
let slackSendPromise: Promise<typeof import("../../slack/send.js")> | null = null;

let telegramMonitorPromise: Promise<typeof import("../../telegram/monitor.js")> | null = null;
let telegramProbePromise: Promise<typeof import("../../telegram/probe.js")> | null = null;
let telegramSendPromise: Promise<typeof import("../../telegram/send.js")> | null = null;

let signalMonitorPromise: Promise<typeof import("../../signal/index.js")> | null = null;
let signalProbePromise: Promise<typeof import("../../signal/probe.js")> | null = null;
let signalSendPromise: Promise<typeof import("../../signal/send.js")> | null = null;

let imessageMonitorPromise: Promise<typeof import("../../imessage/monitor.js")> | null = null;
let imessageProbePromise: Promise<typeof import("../../imessage/probe.js")> | null = null;
let imessageSendPromise: Promise<typeof import("../../imessage/send.js")> | null = null;

let lineMonitorPromise: Promise<typeof import("../../line/monitor.js")> | null = null;
let lineProbePromise: Promise<typeof import("../../line/probe.js")> | null = null;

const runtimeModuleRefs = {
  whatsappLoginTool: {
    srcFile: "src/channels/plugins/agent-tools/whatsapp-login.ts",
    distFile: "dist/channels/plugins/agent-tools/whatsapp-login.js",
  },
  discordActions: {
    srcFile: "src/channels/plugins/actions/discord.ts",
    distFile: "dist/channels/plugins/actions/discord.js",
  },
  signalActions: {
    srcFile: "src/channels/plugins/actions/signal.ts",
    distFile: "dist/channels/plugins/actions/signal.js",
  },
  telegramActions: {
    srcFile: "src/channels/plugins/actions/telegram.ts",
    distFile: "dist/channels/plugins/actions/telegram.js",
  },
  telegramAudit: {
    srcFile: "src/telegram/audit.ts",
    distFile: "dist/telegram/audit.js",
  },
  telegramToken: {
    srcFile: "src/telegram/token.ts",
    distFile: "dist/telegram/token.js",
  },
  lineAccounts: {
    srcFile: "src/line/accounts.ts",
    distFile: "dist/line/accounts.js",
  },
  lineSend: {
    srcFile: "src/line/send.ts",
    distFile: "dist/line/send.js",
  },
  lineTemplateMessages: {
    srcFile: "src/line/template-messages.ts",
    distFile: "dist/line/template-messages.js",
  },
} as const;

function loadWebMediaModule() {
  webMediaPromise ??= import("../../web/media.js");
  return webMediaPromise;
}

function loadWebOutbound() {
  webOutboundPromise ??= import("../../web/outbound.js");
  return webOutboundPromise;
}

function loadWebLogin() {
  webLoginPromise ??= import("../../web/login.js");
  return webLoginPromise;
}

function loadWebLoginQr() {
  webLoginQrPromise ??= import("../../web/login-qr.js");
  return webLoginQrPromise;
}

function loadWebChannel() {
  webChannelPromise ??= import("../../channels/web/index.js");
  return webChannelPromise;
}

function loadWhatsAppActions() {
  whatsappActionsPromise ??= import("../../agents/tools/whatsapp-actions.js");
  return whatsappActionsPromise;
}

function loadWhatsAppAgentToolsSync() {
  whatsappAgentToolsModule ??= requireRuntimeModule<
    typeof import("../../channels/plugins/agent-tools/whatsapp-login.js")
  >(runtimeModuleRefs.whatsappLoginTool);
  return whatsappAgentToolsModule;
}

function loadDiscordActionsSync() {
  discordActionsModule ??= requireRuntimeModule<
    typeof import("../../channels/plugins/actions/discord.js")
  >(runtimeModuleRefs.discordActions);
  return discordActionsModule;
}

function loadSignalActionsSync() {
  signalActionsModule ??= requireRuntimeModule<
    typeof import("../../channels/plugins/actions/signal.js")
  >(runtimeModuleRefs.signalActions);
  return signalActionsModule;
}

function loadTelegramActionsSync() {
  telegramActionsModule ??= requireRuntimeModule<
    typeof import("../../channels/plugins/actions/telegram.js")
  >(runtimeModuleRefs.telegramActions);
  return telegramActionsModule;
}

function loadTelegramAuditSync() {
  telegramAuditModule ??= requireRuntimeModule<typeof import("../../telegram/audit.js")>(
    runtimeModuleRefs.telegramAudit,
  );
  return telegramAuditModule;
}

function loadTelegramTokenSync() {
  telegramTokenModule ??= requireRuntimeModule<typeof import("../../telegram/token.js")>(
    runtimeModuleRefs.telegramToken,
  );
  return telegramTokenModule;
}

function loadLineAccountsSync() {
  lineAccountsModule ??= requireRuntimeModule<typeof import("../../line/accounts.js")>(
    runtimeModuleRefs.lineAccounts,
  );
  return lineAccountsModule;
}

function loadLineSendSync() {
  lineSendModule ??= requireRuntimeModule<typeof import("../../line/send.js")>(
    runtimeModuleRefs.lineSend,
  );
  return lineSendModule;
}

function loadLineTemplateMessagesSync() {
  lineTemplateMessagesModule ??= requireRuntimeModule<
    typeof import("../../line/template-messages.js")
  >(runtimeModuleRefs.lineTemplateMessages);
  return lineTemplateMessagesModule;
}

function loadDiscordAudit() {
  discordAuditPromise ??= import("../../discord/audit.js");
  return discordAuditPromise;
}

function loadDiscordDirectoryLive() {
  discordDirectoryLivePromise ??= import("../../discord/directory-live.js");
  return discordDirectoryLivePromise;
}

function loadDiscordMonitor() {
  discordMonitorPromise ??= import("../../discord/monitor.js");
  return discordMonitorPromise;
}

function loadDiscordProbe() {
  discordProbePromise ??= import("../../discord/probe.js");
  return discordProbePromise;
}

function loadDiscordResolveChannels() {
  discordResolveChannelsPromise ??= import("../../discord/resolve-channels.js");
  return discordResolveChannelsPromise;
}

function loadDiscordResolveUsers() {
  discordResolveUsersPromise ??= import("../../discord/resolve-users.js");
  return discordResolveUsersPromise;
}

function loadDiscordSend() {
  discordSendPromise ??= import("../../discord/send.js");
  return discordSendPromise;
}

function loadSlackActions() {
  slackActionsPromise ??= import("../../agents/tools/slack-actions.js");
  return slackActionsPromise;
}

function loadSlackDirectoryLive() {
  slackDirectoryLivePromise ??= import("../../slack/directory-live.js");
  return slackDirectoryLivePromise;
}

function loadSlackMonitor() {
  slackMonitorPromise ??= import("../../slack/index.js");
  return slackMonitorPromise;
}

function loadSlackProbe() {
  slackProbePromise ??= import("../../slack/probe.js");
  return slackProbePromise;
}

function loadSlackResolveChannels() {
  slackResolveChannelsPromise ??= import("../../slack/resolve-channels.js");
  return slackResolveChannelsPromise;
}

function loadSlackResolveUsers() {
  slackResolveUsersPromise ??= import("../../slack/resolve-users.js");
  return slackResolveUsersPromise;
}

function loadSlackSend() {
  slackSendPromise ??= import("../../slack/send.js");
  return slackSendPromise;
}

function loadTelegramMonitor() {
  telegramMonitorPromise ??= import("../../telegram/monitor.js");
  return telegramMonitorPromise;
}

function loadTelegramProbe() {
  telegramProbePromise ??= import("../../telegram/probe.js");
  return telegramProbePromise;
}

function loadTelegramSend() {
  telegramSendPromise ??= import("../../telegram/send.js");
  return telegramSendPromise;
}

function loadSignalMonitor() {
  signalMonitorPromise ??= import("../../signal/index.js");
  return signalMonitorPromise;
}

function loadSignalProbe() {
  signalProbePromise ??= import("../../signal/probe.js");
  return signalProbePromise;
}

function loadSignalSend() {
  signalSendPromise ??= import("../../signal/send.js");
  return signalSendPromise;
}

function loadIMessageMonitor() {
  imessageMonitorPromise ??= import("../../imessage/monitor.js");
  return imessageMonitorPromise;
}

function loadIMessageProbe() {
  imessageProbePromise ??= import("../../imessage/probe.js");
  return imessageProbePromise;
}

function loadIMessageSend() {
  imessageSendPromise ??= import("../../imessage/send.js");
  return imessageSendPromise;
}

function loadLineMonitor() {
  lineMonitorPromise ??= import("../../line/monitor.js");
  return lineMonitorPromise;
}

function loadLineProbe() {
  lineProbePromise ??= import("../../line/probe.js");
  return lineProbePromise;
}

export function createPluginRuntime(): PluginRuntime {
  return {
    version: resolveVersion(),
    config: createRuntimeConfig(),
    system: createRuntimeSystem(),
    media: createRuntimeMedia(),
    tts: { textToSpeechTelephony },
    tools: createRuntimeTools(),
    channel: createRuntimeChannel(),
    logging: createRuntimeLogging(),
    state: { resolveStateDir },
  };
}

function createRuntimeConfig(): PluginRuntime["config"] {
  return {
    loadConfig,
    writeConfigFile,
  };
}

function createRuntimeSystem(): PluginRuntime["system"] {
  return {
    enqueueSystemEvent,
    runCommandWithTimeout,
    formatNativeDependencyHint,
  };
}

function createRuntimeMedia(): PluginRuntime["media"] {
  return {
    loadWebMedia: loadWebMediaLazy,
    detectMime,
    mediaKindFromMime,
    isVoiceCompatibleAudio,
    getImageMetadata,
    resizeToJpeg,
  };
}

function createRuntimeTools(): PluginRuntime["tools"] {
  return {
    createMemoryGetTool,
    createMemorySearchTool,
    registerMemoryCli,
  };
}

function createRuntimeChannel(): PluginRuntime["channel"] {
  return {
    text: {
      chunkByNewline,
      chunkMarkdownText,
      chunkMarkdownTextWithMode,
      chunkText,
      chunkTextWithMode,
      resolveChunkMode,
      resolveTextChunkLimit,
      hasControlCommand,
      resolveMarkdownTableMode,
      convertMarkdownTables,
    },
    reply: {
      dispatchReplyWithBufferedBlockDispatcher,
      createReplyDispatcherWithTyping,
      resolveEffectiveMessagesConfig,
      resolveHumanDelayConfig,
      dispatchReplyFromConfig,
      withReplyDispatcher,
      finalizeInboundContext,
      formatAgentEnvelope,
      /** @deprecated Prefer `BodyForAgent` + structured user-context blocks (do not build plaintext envelopes for prompts). */
      formatInboundEnvelope,
      resolveEnvelopeFormatOptions,
    },
    routing: {
      resolveAgentRoute,
    },
    pairing: {
      buildPairingReply,
      readAllowFromStore: ({ channel, accountId, env }) =>
        readChannelAllowFromStore(channel, env, accountId),
      upsertPairingRequest: ({ channel, id, accountId, meta, env, pairingAdapter }) =>
        upsertChannelPairingRequest({
          channel,
          id,
          accountId,
          meta,
          env,
          pairingAdapter,
        }),
    },
    media: {
      fetchRemoteMedia,
      saveMediaBuffer,
    },
    activity: {
      record: recordChannelActivity,
      get: getChannelActivity,
    },
    session: {
      resolveStorePath,
      readSessionUpdatedAt,
      recordSessionMetaFromInbound,
      recordInboundSession,
      updateLastRoute,
    },
    mentions: {
      buildMentionRegexes,
      matchesMentionPatterns,
      matchesMentionWithExplicit,
    },
    reactions: {
      shouldAckReaction,
      removeAckReactionAfterReply,
    },
    groups: {
      resolveGroupPolicy: resolveChannelGroupPolicy,
      resolveRequireMention: resolveChannelGroupRequireMention,
    },
    debounce: {
      createInboundDebouncer,
      resolveInboundDebounceMs,
    },
    commands: {
      resolveCommandAuthorizedFromAuthorizers,
      isControlCommandMessage,
      shouldComputeCommandAuthorized,
      shouldHandleTextCommands,
    },
    discord: {
      messageActions: discordMessageActionsLazy,
      auditChannelPermissions: auditDiscordChannelPermissionsLazy,
      listDirectoryGroupsLive: listDiscordDirectoryGroupsLiveLazy,
      listDirectoryPeersLive: listDiscordDirectoryPeersLiveLazy,
      probeDiscord: probeDiscordLazy,
      resolveChannelAllowlist: resolveDiscordChannelAllowlistLazy,
      resolveUserAllowlist: resolveDiscordUserAllowlistLazy,
      sendMessageDiscord: sendMessageDiscordLazy,
      sendPollDiscord: sendPollDiscordLazy,
      monitorDiscordProvider: monitorDiscordProviderLazy,
    },
    slack: {
      listDirectoryGroupsLive: listSlackDirectoryGroupsLiveLazy,
      listDirectoryPeersLive: listSlackDirectoryPeersLiveLazy,
      probeSlack: probeSlackLazy,
      resolveChannelAllowlist: resolveSlackChannelAllowlistLazy,
      resolveUserAllowlist: resolveSlackUserAllowlistLazy,
      sendMessageSlack: sendMessageSlackLazy,
      monitorSlackProvider: monitorSlackProviderLazy,
      handleSlackAction: handleSlackActionLazy,
    },
    telegram: {
      auditGroupMembership: auditTelegramGroupMembershipLazy,
      collectUnmentionedGroupIds: collectTelegramUnmentionedGroupIdsLazy,
      probeTelegram: probeTelegramLazy,
      resolveTelegramToken: resolveTelegramTokenLazy,
      sendMessageTelegram: sendMessageTelegramLazy,
      sendPollTelegram: sendPollTelegramLazy,
      monitorTelegramProvider: monitorTelegramProviderLazy,
      messageActions: telegramMessageActionsLazy,
    },
    signal: {
      probeSignal: probeSignalLazy,
      sendMessageSignal: sendMessageSignalLazy,
      monitorSignalProvider: monitorSignalProviderLazy,
      messageActions: signalMessageActionsLazy,
    },
    imessage: {
      monitorIMessageProvider: monitorIMessageProviderLazy,
      probeIMessage: probeIMessageLazy,
      sendMessageIMessage: sendMessageIMessageLazy,
    },
    whatsapp: {
      getActiveWebListener,
      getWebAuthAgeMs,
      logoutWeb,
      logWebSelfId,
      readWebSelfId,
      webAuthExists,
      sendMessageWhatsApp: sendMessageWhatsAppLazy,
      sendPollWhatsApp: sendPollWhatsAppLazy,
      loginWeb: loginWebLazy,
      startWebLoginWithQr: startWebLoginWithQrLazy,
      waitForWebLogin: waitForWebLoginLazy,
      monitorWebChannel: monitorWebChannelLazy,
      handleWhatsAppAction: handleWhatsAppActionLazy,
      createLoginTool: createWhatsAppLoginToolLazy,
    },
    line: {
      listLineAccountIds: listLineAccountIdsLazy,
      resolveDefaultLineAccountId: resolveDefaultLineAccountIdLazy,
      resolveLineAccount: resolveLineAccountLazy,
      normalizeAccountId: normalizeLineAccountIdLazy,
      probeLineBot: probeLineBotLazy,
      sendMessageLine: sendMessageLineLazy,
      pushMessageLine: pushMessageLineLazy,
      pushMessagesLine: pushMessagesLineLazy,
      pushFlexMessage: pushFlexMessageLazy,
      pushTemplateMessage: pushTemplateMessageLazy,
      pushLocationMessage: pushLocationMessageLazy,
      pushTextMessageWithQuickReplies: pushTextMessageWithQuickRepliesLazy,
      createQuickReplyItems: createQuickReplyItemsLazy,
      buildTemplateMessageFromPayload: buildTemplateMessageFromPayloadLazy,
      monitorLineProvider: monitorLineProviderLazy,
    },
  };
}

function createRuntimeLogging(): PluginRuntime["logging"] {
  return {
    shouldLogVerbose,
    getChildLogger: (bindings, opts) => {
      const logger = getChildLogger(bindings, {
        level: opts?.level ? normalizeLogLevel(opts.level) : undefined,
      });
      return {
        debug: (message) => logger.debug?.(message),
        info: (message) => logger.info(message),
        warn: (message) => logger.warn(message),
        error: (message) => logger.error(message),
      };
    },
  };
}

export type { PluginRuntime } from "./types.js";
