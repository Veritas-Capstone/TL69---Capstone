type FetchOptions = RequestInit & {
	timeoutMs?: number;
	retries?: number;
	retryDelayMs?: number;
};

function sleep(ms: number) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status: number) {
	return status >= 500 || status === 408 || status === 429;
}

export async function fetchWithTimeout(
	url: string,
	options: FetchOptions = {},
): Promise<Response> {
	const {
		timeoutMs = 20_000,
		retries = 1,
		retryDelayMs = 500,
		...init
	} = options;

	let lastError: unknown;

	for (let attempt = 0; attempt <= retries; attempt++) {
		const controller = new AbortController();
		const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

		try {
			const response = await fetch(url, {
				...init,
				signal: controller.signal,
			});

			if (!response.ok && isRetryableStatus(response.status) && attempt < retries) {
				await sleep(retryDelayMs * (attempt + 1));
				continue;
			}

			return response;
		} catch (err) {
			lastError = err;
			const isAbort = err instanceof DOMException && err.name === 'AbortError';
			if (attempt < retries && !isAbort) {
				await sleep(retryDelayMs * (attempt + 1));
				continue;
			}
			throw err;
		} finally {
			clearTimeout(timeoutId);
		}
	}

	throw lastError;
}
