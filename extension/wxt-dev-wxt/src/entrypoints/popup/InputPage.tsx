import { Button } from '@/components/ui/button';
import { CardDescription, CardContent, Card } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { useState } from 'react';
import { GlobeIcon, ScanIcon, SearchIcon } from 'lucide-react';

export default function InputPage({
	setText,
	callModel,
}: {
	setText: React.Dispatch<React.SetStateAction<string | undefined>>;
	callModel: Function;
}) {
	const [tempText, setTempText] = useState<string>();

	// selects text from entire webpage
	async function scanEntirePage() {
		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;

		await browser.tabs.sendMessage(tab.id, { type: 'GET_PAGE_TEXT' });
		callModel();
	}

	async function analyzeText() {
		setText(tempText);
		await browser.storage.local.set({ selectedText: tempText });
		await callModel();
	}

	return (
		<>
			<CardDescription className="text-xs text-gray-500">
				Detect bias, verify facts, find perspectives
			</CardDescription>
			<CardContent className="w-full flex flex-col gap-4">
				<Card className="bg-gray-100">
					<CardContent className="flex flex-col gap-4">
						<div className="flex gap-2">
							<div className="flex items-center justify-center bg-gray-300 rounded-md min-w-9">
								<GlobeIcon />
							</div>
							<div>
								<div className="flex flex-col">
									<h3 className="text-sm">Analyze This Page</h3>
									<p className="text-xs text-gray-500">Scan the current webpage</p>
								</div>
							</div>
						</div>
						<Button className="text-xs" onClick={scanEntirePage}>
							<ScanIcon /> Scan Current Page
						</Button>
					</CardContent>
				</Card>
				<div className="flex items-center gap-4 my-2">
					<Separator className="flex-1" />
					<span className="text-muted-foreground">or paste text</span>
					<Separator className="flex-1" />
				</div>
				<Textarea
					onChange={(e) => setTempText(e.target.value)}
					placeholder="Paste text content here..."
					className="max-h-[50px] bg-gray-100 text-sm"
				/>
				<Button variant={'outline'} disabled={!tempText} onClick={analyzeText}>
					<SearchIcon /> Analyze Text
				</Button>
			</CardContent>
		</>
	);
}
