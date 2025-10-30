// sample fetch
export default async function fetchAPI(text: string) {
	console.log(text);
	const response = await fetch(`https://dummyjson.com/http/200?delay=5000`, {
		headers: { 'Content-Type': 'application/json' },
		method: 'POST',
		body: JSON.stringify({
			text: text,
		}),
	});

	if (response.status === 200) {
		const data = await response.json();
		return data.message;
	} else {
		// await throwAPIError(response)
	}
}
