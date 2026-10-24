const APP_CONFIG = Object.freeze({
  APP_TITLE: 'ScopeGuard',

  // Paste the two exact values printed by the final Colab launch cell.
  // Use the base URL only. Do not add /health or /process here.
  COLAB_API_BASE_URL: 'https://YOUR-MODEL-BACKEND.example',
  COLAB_API_SECRET: 'REDACTED_BEFORE_SUBMISSION',

  MODEL_INSTRUCTION: [
    'Compare the provided Project Scope with the New Request and determine whether the request fits the stated project requirements.',
    'Return four labeled sections: Verdict, Reason, Missing Information, and Recommended Action.',
    'The Verdict must be exactly one of: IN SCOPE, OUT OF SCOPE, or NEEDS CLARIFICATION.',
    'If important information is missing or ambiguous, use NEEDS CLARIFICATION and identify what information is needed.',
    'Base the classification only on the Project Scope and New Request provided by the user.',
    'Do not invent project requirements, assumptions, deadlines, features, constraints, or other missing details.'
  ].join(' '),

  MIN_WORDS: 10,
  MAX_CHARACTERS: 5000,
  MAX_NEW_TOKENS: 220
});

function doGet() {
  return HtmlService.createTemplateFromFile('Index')
    .evaluate()
    .setTitle(APP_CONFIG.APP_TITLE);
}

function getPublicConfig() {
  return {
    appTitle: APP_CONFIG.APP_TITLE,
    minWords: APP_CONFIG.MIN_WORDS,
    maxCharacters: APP_CONFIG.MAX_CHARACTERS
  };
}

function testModelConnection() {
  const configurationError = validateConfiguration_();

  if (configurationError) {
    return failure_(configurationError, 'CONFIGURATION_ERROR');
  }

  try {
    const response = UrlFetchApp.fetch(endpoint_('health'), {
      method: 'get',
      muteHttpExceptions: true
    });

    const status = response.getResponseCode();
    const parsed = parseJson_(response.getContentText());

    if (status !== 200 || !parsed || !parsed.ok) {
      return failure_(
        `The Colab backend responded with status ${status}.`,
        'BACKEND_NOT_READY'
      );
    }

    return {
      ok: true,
      message: `Connected to ${parsed.model} on ${parsed.device}.`,
      model: parsed.model,
      device: parsed.device
    };

  } catch (error) {
    console.error(error);

    return failure_(
      'Could not reach the verified Colab backend. Confirm that the notebook runtime and tunnel are still running.',
      'BACKEND_UNAVAILABLE'
    );
  }
}

function processProductivityText(payload) {
  const started = Date.now();

  const inputResult = validateInput_(payload && payload.text);

  if (!inputResult.ok) {
    return failure_(
      inputResult.message,
      'VALIDATION_ERROR'
    );
  }

  const configurationError = validateConfiguration_();

  if (configurationError) {
    return failure_(
      configurationError,
      'CONFIGURATION_ERROR'
    );
  }

  try {
    const modelResult = callColabModel_(inputResult.cleaned);

    const warnings = checkOutput_(
      inputResult.cleaned,
      modelResult.output
    );

    return {
      ok: true,
      output: modelResult.output,
      warnings: warnings,
      model: modelResult.model || 'Pretrained text model',
      device: modelResult.device || 'Unknown',
      inferenceMs: modelResult.inference_ms || null,
      totalMs: Date.now() - started,
      requiresHumanReview: true
    };

  } catch (error) {
    console.error(error);

    return failure_(
      'The model backend is unavailable or could not complete the request. Confirm that Colab still reports BACKEND READY. No action was taken.',
      'MODEL_SERVICE_ERROR'
    );
  }
}

function validateInput_(value) {
  const cleaned = String(value || '')
    .replace(/\s+/g, ' ')
    .trim();

  if (!cleaned) {
    return {
      ok: false,
      message: 'Enter text before running the tool.'
    };
  }

  const wordCount = cleaned.split(/\s+/).length;

  if (wordCount < APP_CONFIG.MIN_WORDS) {
    return {
      ok: false,
      message: `Provide at least ${APP_CONFIG.MIN_WORDS} words so the model has enough context.`
    };
  }

  if (cleaned.length > APP_CONFIG.MAX_CHARACTERS) {
    return {
      ok: false,
      message: `Input must contain no more than ${APP_CONFIG.MAX_CHARACTERS.toLocaleString()} characters.`
    };
  }

  // ScopeGuard-specific validation
  const lowerCleaned = cleaned.toLowerCase();

  if (
    !lowerCleaned.includes('project scope:') ||
    !lowerCleaned.includes('new request:')
  ) {
    return {
      ok: false,
      message: 'Input must include both "Project Scope:" and "New Request:" sections.'
    };
  }

  return {
    ok: true,
    cleaned: cleaned
  };
}

function validateConfiguration_() {
  if (
    !APP_CONFIG.COLAB_API_BASE_URL.startsWith('https://') ||
    APP_CONFIG.COLAB_API_BASE_URL.includes('PASTE_')
  ) {
    return 'Paste the verified Colab base URL into COLAB_API_BASE_URL in Code.gs.';
  }

  if (
    !APP_CONFIG.COLAB_API_SECRET ||
    APP_CONFIG.COLAB_API_SECRET.includes('PASTE_')
  ) {
    return 'Paste the current Colab secret into COLAB_API_SECRET in Code.gs.';
  }

  return null;
}

function endpoint_(path) {
  return APP_CONFIG.COLAB_API_BASE_URL.replace(/\/+$/, '') + '/' + path;
}

function callColabModel_(text) {
  const response = UrlFetchApp.fetch(endpoint_('process'), {
    method: 'post',
    contentType: 'application/json',

    headers: {
      'X-Build-Secret': APP_CONFIG.COLAB_API_SECRET
    },

    payload: JSON.stringify({
      text: text,
      instruction: APP_CONFIG.MODEL_INSTRUCTION,
      max_new_tokens: APP_CONFIG.MAX_NEW_TOKENS
    }),

    muteHttpExceptions: true
  });

  const status = response.getResponseCode();
  const parsed = parseJson_(response.getContentText());

  if (!parsed) {
    throw new Error(
      `The model backend returned an unreadable response (${status}).`
    );
  }

  if (
    status < 200 ||
    status >= 300 ||
    !parsed.ok
  ) {
    throw new Error(
      parsed.error ||
      `The model backend returned status ${status}.`
    );
  }

  if (!String(parsed.output || '').trim()) {
    throw new Error(
      'The model returned an empty result.'
    );
  }

  return parsed;
}

function parseJson_(value) {
  try {
    return JSON.parse(value);
  } catch (error) {
    return null;
  }
}

function checkOutput_(sourceText, modelOutput) {
  const warnings = [];

  const source = sourceText.toLowerCase();
  const output = String(modelOutput || '').trim();
  const upperOutput = output.toUpperCase();

  if (output.length < 20) {
    warnings.push(
      'The model response is unusually short. Verify that it is complete.'
    );
  }

  const uncertainTerms = [
    'may',
    'might',
    'maybe',
    'possibly',
    'probably',
    'soon',
    'later',
    'next week',
    'sometime'
  ];

  const foundTerms = uncertainTerms.filter(term =>
    source.includes(term)
  );

  if (foundTerms.length) {
    warnings.push(
      `The source contains uncertain language: ${foundTerms.join(', ')}. Confirm the scope classification and project requirements before approval.`
    );
  }

  if (
    source.includes('not selected') ||
    source.includes('not decided')
  ) {
    warnings.push(
      'The source contains an unresolved decision that requires human judgment.'
    );
  }

  // ScopeGuard-specific output validation
  const validVerdicts = [
    'IN SCOPE',
    'OUT OF SCOPE',
    'NEEDS CLARIFICATION'
  ];

  const hasValidVerdict = validVerdicts.some(verdict =>
    upperOutput.includes(verdict)
  );

  if (!hasValidVerdict) {
    warnings.push(
      'The model output does not contain a valid ScopeGuard verdict. Review before approval.'
    );
  }

  // Visible ambiguity handling
  if (upperOutput.includes('NEEDS CLARIFICATION')) {
    warnings.push(
      'ScopeGuard identified missing or ambiguous information. Clarify the project requirements before approving the decision.'
    );
  }

  return warnings;
}

function failure_(message, code) {
  return {
    ok: false,
    error: message,
    code: code,
    requiresHumanReview: false
  };
}
