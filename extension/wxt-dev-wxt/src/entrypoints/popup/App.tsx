import { useState, useEffect } from 'react';
import fetchAPI from './fetchAPI';
import '@/assets/tailwind.css';
import { Card, CardDescription, CardHeader } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import InputPage from './InputPage';
import AnalysisPage from './AnalysisPage';

function App() {
	const [text, setText] = useState<string>();
	const [result, setResult] = useState<{ bias: string; validity: string }>();

	async function callModel() {
		const text = await browser.storage.local.get('selectedText');
		await browser.storage.local.clear();
		setText(text.selectedText);
		const data = await fetchAPI(text.selectedText);
		setResult({ bias: 'Center', validity: 'Factual' });
	}

	useEffect(() => {
		setText('');

		callModel();
	}, []);

	return (
		<>
			<Card className="rounded-none min-w-[350px] p-0 flex flex-col items-center gap-4 shadow-none border-b-0">
				<CardHeader className="from-gray-900 to-gray-800 gap-0 py-2 w-full bg-linear-to-r">
					<h1 className="font-semibold text-xl text-white">Veritas</h1>
				</CardHeader>
				{!text ? (
					<InputPage setText={setText} callModel={callModel} />
				) : (
					<AnalysisPage text={text} setText={setText} />
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
