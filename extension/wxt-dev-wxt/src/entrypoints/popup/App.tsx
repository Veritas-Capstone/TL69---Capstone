import { useState, useEffect } from 'react';
import fetchAPI from './fetchAPI';
import '@/assets/tailwind.css';

function App() {
	const [text, setText] = useState('');
	const [result, setResult] = useState<{ bias: string; validity: string }>();

	useEffect(() => {
		setText('');
		async function callModel() {
			const text = await browser.storage.local.get('selectedText');
			await browser.storage.local.clear();
			setText(text.selectedText);
			const data = await fetchAPI(text.selectedText);
			setResult({ bias: 'Center', validity: 'Factual' });
		}

		callModel();
	}, []);

	return (
		<>
			<h1>Veritas</h1>
			<h2>Selected text:</h2>
			<p>{text ?? 'No text selected'}</p>
			{text && (
				<>
					<h2>Results</h2>
					{result ? (
						<p>
							Bias: {result.bias}, Validity: {result.validity}
						</p>
					) : (
						<p>{text ? 'Loading...' : ''}</p>
					)}
				</>
			)}
		</>
	);
}

export default App;
