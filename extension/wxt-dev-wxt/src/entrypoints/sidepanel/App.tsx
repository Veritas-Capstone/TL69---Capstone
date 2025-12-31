import { useState, useEffect } from 'react';
import fetchAPI from './fetchAPI';
import '@/assets/tailwind.css';
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import InputPage from './InputPage';
import AnalysisPage from './AnalysisPage';
import { Spinner } from '@/components/ui/spinner';
import { AnalysisResult } from '@/types';

function App() {
	const [text, setText] = useState<string>();
	const [result, setResult] = useState<AnalysisResult | undefined>();
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string>();

	// call model on selected text
	async function callModel() {
		const selectedText = await browser.storage.local.get('selectedText');
		setText(selectedText.selectedText);

		setIsLoading(true);
		setError(undefined);

		try {
			const [tab] = await browser.tabs.query({ active: true, currentWindow: true });

			// clear selected text on webpage
			await browser.tabs.sendMessage(tab.id ?? 0, { type: 'CLEAR_SELECTION' });

			// call model API
			const data = await fetchAPI(selectedText.selectedText);
			setResult(data);

			// underline claims on webpage
			await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				sentences: data?.bias_claims,
			});
		} catch (err) {
			setError('Failed to analyze text. Make sure the backend server is running on http://localhost:8000');
		} finally {
			setIsLoading(false);
		}
	}

	useEffect(() => {
		// call model on selected text when needed
		const handleCallModel = (message: any) => {
			if (message.type === 'CALL_MODEL') {
				callModel();
			}
		};

		callModel();
		browser.runtime.onMessage.addListener(handleCallModel);
		return () => browser.runtime.onMessage.removeListener(handleCallModel);
	}, []);

	return (
		<>
			<Card className="rounded-none min-w-[300px] flex-1 overflow-y-auto p-0 flex flex-col items-center gap-4 shadow-none border-b-0">
				<CardHeader className="from-gray-900 to-gray-800 gap-0 py-2 w-full bg-linear-to-r rounded-tl-xl">
					<h1 className="font-semibold text-xl text-white">Veritas</h1>
				</CardHeader>
				{error && (
					<CardContent className="w-full">
						<div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-800">{error}</div>
					</CardContent>
				)}
				{isLoading ? (
					<CardContent className="w-full h-full flex justify-center items-center gap-2">
						<Spinner className="w-5 h-5" />
						<p className="text-base">Analyzing Text...</p>
					</CardContent>
				) : !result ? (
					<InputPage setText={setText} callModel={callModel} />
				) : (
					<AnalysisPage text={text} setText={setText} result={result} setResult={setResult} />
				)}
			</Card>
			<div className="flex flex-col justify-center gap-2 mt-8">
				<Separator className="mt-2" />
				<CardDescription className="mx-auto mb-3 text-xs">Powered by Veritas AI</CardDescription>
			</div>
		</>
	);
}

export default App;
