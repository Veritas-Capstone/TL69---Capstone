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
		return {
			checks: 3,
			issues: 2,
			claims: [
				{
					text: 'Freedom House published a report Wednesday downgrading the United States from a democracy to whatever political system lobsters have.',
					category: 'Unverified Source',
					description: 'Quotes or attributions lack confirmation from credible or official sources.',
					valid: false,
				},
				{
					text: 'Persistent executive overreach and erosion of civil liberties mean that America now looks less like a traditional federal republic',
					category: 'Accurate Contextual Information',
					description: 'Statement is accurate and supported by credible, verifiable information.',
					valid: true,
				},
				{
					text: 'Our nation already passed the tipping point where we might hope to match the deliberative bicameral legislative process of, say, shore crabs. At this juncture, there’s just too much scuttling in American politics to call it anything other than a flawed lobster republic.',
					category: 'False or Misleading',
					description: 'Claims are not supported by verifiable sources or evidence.',
					valid: false,
				},
			],
		};
	} else {
		// await throwAPIError(response)
	}
}
