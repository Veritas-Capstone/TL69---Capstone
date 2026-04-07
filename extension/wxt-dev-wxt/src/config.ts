// Centralized backend URLs for the extension.
// Change these in extension/wxt-dev-wxt/.env for local/dev overrides.

export const MODEL_BACKEND =
	import.meta.env.WXT_MODEL_BACKEND || 'http://localhost:8000';

export const USER_AUTH_BACKEND =
	import.meta.env.WXT_USER_AUTH_BACKEND || 'http://localhost:8080';
