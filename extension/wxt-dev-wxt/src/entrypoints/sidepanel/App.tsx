import { useState, useEffect } from 'react';
import fetchAPI from './fetchAPI';
import '@/assets/tailwind.css';
import { Card, CardContent, CardDescription, CardFooter, CardHeader } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import InputPage from './InputPage';
import AnalysisPage from './AnalysisPage';
import { Spinner } from '@/components/ui/spinner';
import { AnalysisResult } from '@/types';
import { Button } from '@/components/ui/button';
import Profile from './Profile';
import { ArrowLeftIcon } from 'lucide-react';

function App() {
	const [text, setText] = useState<string>();
	const [result, setResult] = useState<AnalysisResult | undefined>();
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string>();
	const [failedUnderlinesArr, setFailedUnderlinesArr] = useState<number[]>([]);
	const [profile, setProfile] = useState(false);

	// call model on selected text
	async function callModel() {
		const selectedText = await browser.storage.local.get('selectedText');
		if (!selectedText.selectedText) {
			return;
		}

		setText(selectedText.selectedText);
		setIsLoading(true);
		setError(undefined);

		try {
			const [tab] = await browser.tabs.query({ active: true, currentWindow: true });

			// clear selected text on webpage
			await browser.tabs.sendMessage(tab.id ?? 0, { type: 'CLEAR_SELECTION' });

			// call model API
			const storedResult = await browser.storage.local.get('storedResult');
			let data: AnalysisResult;
			if (!storedResult.storedResult) {
				data = await fetchAPI(selectedText.selectedText);
				await browser.storage.local.set({ storedResult: data });
			} else {
				data = storedResult.storedResult;
			}
			setResult(data);

			// underline claims on webpage
			const failed = await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				sentences: data?.bias_claims,
			});

			// update user stats
			if (localStorage.getItem('username')) {
				await fetch(`http://localhost:8080/stats`, {
					headers: { 'Content-Type': 'application/json' },
					method: 'POST',
					body: JSON.stringify({
						username: localStorage.getItem('username'),
						leftBias:
							data?.bias_claims.filter((x, idx) => x.category === 'Left-leaning' && !failed.includes(idx))
								.length ?? 0,
						rightBias:
							data?.bias_claims.filter((x, idx) => x.category === 'Right-leaning' && !failed.includes(idx))
								.length ?? 0,
						centerBias:
							data?.bias_claims.filter((x, idx) => x.category === 'Centrist' && !failed.includes(idx))
								.length ?? 0,
					}),
				});
			}
			setFailedUnderlinesArr(failed);
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
		<Card className="rounded-none w-full h-full flex-1 overflow-y-auto p-0 flex flex-col items-center gap-4 shadow-none border-b-0">
			<CardHeader className="from-gray-900 to-gray-800 gap-0 py-2 w-full bg-linear-to-r rounded-tl-xl">
				<div className="flex items-center justify-between">
					<h1 className="font-semibold text-xl text-white">Veritas</h1>
					{!profile ? (
						<Button className="text-white" variant={'link'} onClick={() => setProfile(true)}>
							Profile
						</Button>
					) : (
						<Button className="text-white" variant={'link'} onClick={() => setProfile(false)}>
							Back
						</Button>
					)}
				</div>
			</CardHeader>
			<CardContent className="w-full flex-1 flex flex-col gap-4">
				{isLoading ? (
					<div className="w-full flex-1 flex justify-center items-center gap-2">
						<Spinner className="w-5 h-5" />
						<p className="text-base">Analyzing Text...</p>
					</div>
				) : (
					<>
						{error && (
							<div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-800">{error}</div>
						)}
						{profile ? (
							<Profile />
						) : !result ? (
							<InputPage setText={setText} callModel={callModel} />
						) : (
							<AnalysisPage
								text={text}
								setText={setText}
								result={result}
								setResult={setResult}
								failedUnderlinesArr={failedUnderlinesArr}
								setFailedUnderlinesArr={setFailedUnderlinesArr}
							/>
						)}
					</>
				)}
			</CardContent>
			<CardFooter className="flex flex-col w-full gap-2"></CardFooter>
		</Card>
	);
}

export default App;
