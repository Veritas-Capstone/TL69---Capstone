import { useState, useEffect } from 'react';
import fetchAPI from './fetchAPI';
import '@/assets/tailwind.css';
import { Card, CardContent, CardFooter, CardHeader } from '@/components/ui/card';
import InputPage from './InputPage';
import AnalysisPage from './AnalysisPage';
import { Spinner } from '@/components/ui/spinner';
import { AnalysisResult } from '@/types';
import { Button } from '@/components/ui/button';
import Profile from './Profile';
import logo from '../../assets/logo.png';

function App() {
	const modelBackend = import.meta.env.WXT_MODEL_BACKEND || 'http://localhost:8000';
	const [result, setResult] = useState<AnalysisResult | undefined>();
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string>();
	const [failedUnderlinesArr, setFailedUnderlinesArr] = useState<number[]>([]);
	const [profile, setProfile] = useState(false);
	const [currentTab, setCurrentTab] = useState<string>('bias');

	// call model on selected text
	async function callModel() {
		const selectedText = await browser.storage.local.get('selectedText');
		if (!selectedText.selectedText) {
			return;
		}

		await browser.storage.local.clear();
		setIsLoading(true);
		setError(undefined);

		try {
			const [tab] = await browser.tabs.query({ active: true, currentWindow: true });

			// clear selected text on webpage
			try {
				await browser.tabs.sendMessage(tab.id ?? 0, { type: 'CLEAR_SELECTION' });
			} catch {
				// Content script not available on this page, that's fine
			}

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
			let failed: number[] = [];
			try {
				failed = await browser.tabs.sendMessage(tab.id ?? 0, {
					type: 'UNDERLINE_SELECTION',
					sentences: data?.bias_claims,
				});
			} catch {
				// Content script unavailable
			}

			await updateStats(data);
			setFailedUnderlinesArr(failed);
		} catch (err) {
			setError(`Failed to analyze text. Make sure the backend server is running on ${modelBackend}`);
		} finally {
			setIsLoading(false);
		}
	}

	// update user stats in profile
	async function updateStats(data: AnalysisResult) {
		if (localStorage.getItem('username')) {
			await fetch(`${import.meta.env.WXT_USER_AUTH_BACKEND}/stats`, {
				headers: { 'Content-Type': 'application/json' },
				method: 'POST',
				body: JSON.stringify({
					username: localStorage.getItem('username'),
					leftBias: data?.bias_claims.filter((x) => x.category === 'Left-leaning').length ?? 0,
					rightBias: data?.bias_claims.filter((x) => x.category === 'Right-leaning').length ?? 0,
					centerBias:
						data?.bias_claims.filter((x) => x.category === 'Centrist' || x.category === 'Neutral/Balanced')
							.length ?? 0,
					supportedClaim: data?.fact_check_claims.filter((x) => x.label === 'SUPPORTED').length ?? 0,
					refutedClaim: data?.fact_check_claims.filter((x) => x.label === 'REFUTED').length ?? 0,
					noInfoClaim: data?.fact_check_claims.filter((x) => x.label === 'NOT ENOUGH INFO').length ?? 0,
				}),
			});
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
		<Card className="font-nunito rounded-none w-full h-full flex-1 overflow-y-auto p-0 flex flex-col items-center gap-4 shadow-none border-b-0">
			<CardHeader className="bg-black gap-0 py-2 w-full bg-linear-to-r rounded-tl-xl">
				<div className="flex items-center justify-between h-full">
					<img src={logo} className="h-[25px] w-[90px]" />
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
							<InputPage callModel={callModel} />
						) : (
							<AnalysisPage
								result={result}
								setResult={setResult}
								failedUnderlinesArr={failedUnderlinesArr}
								setFailedUnderlinesArr={setFailedUnderlinesArr}
								currentTab={currentTab}
								setCurrentTab={setCurrentTab}
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
