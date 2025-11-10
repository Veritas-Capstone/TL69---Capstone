import { useState, useEffect } from 'react';
import fetchAPI from './fetchAPI';
import '@/assets/tailwind.css';
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import InputPage from './InputPage';
import AnalysisPage from './AnalysisPage';
import { Spinner } from '@/components/ui/spinner';

function App() {
	const [text, setText] = useState<string>();
	const [result, setResult] = useState<
		| {
				checks: number;
				issues: number;
				claims: { text: string; category: string; description: string; valid: boolean }[];
		  }
		| undefined
	>();
	const [isLoading, setIsLoading] = useState(false);

	// call model on selected text
	// text can be selected manually through highlighting -> right click menu
	// or by selecting "scan entire webpage" button in popup
	async function callModel() {
		const stored = await browser.storage.local.get('result');
		const selectedText = await browser.storage.local.get('selectedText');
		setText(selectedText.selectedText);

		if (Object.keys(stored).length === 0 && selectedText.selectedText) {
			setIsLoading(true);
			const data = await fetchAPI(selectedText.selectedText);
			setResult(data);
			await browser.storage.local.set({ result: data });

			// underline claims on webpage
			const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
			if (!tab?.id) {
				return;
			}
			// "targets" are sentences that will be highlighted
			await browser.tabs.sendMessage(tab.id, {
				type: 'UNDERLINE_SELECTION_INVALID',
				targets: data?.claims.filter((x) => !x.valid).map((x) => x.text),
			});
			await browser.tabs.sendMessage(tab.id, {
				type: 'UNDERLINE_SELECTION_VALID',
				targets: data?.claims.filter((x) => x.valid).map((x) => x.text),
			});
			window.getSelection()?.removeAllRanges();
			setIsLoading(false);
		} else {
			setResult(stored.result);
		}
	}

	useEffect(() => {
		// call model on selected text when popup opened
		callModel();
	}, []);

	return (
		<>
			<Card className="rounded-none min-w-[300px] overflow-y-auto p-0 flex flex-col items-center gap-4 shadow-none border-b-0">
				<CardHeader className="from-gray-900 to-gray-800 gap-0 py-2 w-full bg-linear-to-r">
					<h1 className="font-semibold text-xl text-white">Veritas</h1>
				</CardHeader>
				{isLoading ? (
					<CardContent className="w-full h-[100px] flex justify-center items-center gap-2">
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
