import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList } from '@/components/ui/tabs';
import { TabsTrigger } from '@radix-ui/react-tabs';
import { TextIcon } from 'lucide-react';
import BiasTab from './BiasTab';
import ClaimTab from './ClaimTab';
import { AnalysisResult } from '@/types';

type AnalysisProps = {
	text: string | undefined;
	setText: React.Dispatch<React.SetStateAction<string | undefined>>;
	result: AnalysisResult | undefined;
	setResult: React.Dispatch<React.SetStateAction<AnalysisResult | undefined>>;
	failedUnderlinesArr: number[];
	setFailedUnderlinesArr: React.Dispatch<React.SetStateAction<number[]>>;
};

export default function AnalysisPage({
	text,
	setText,
	result,
	setResult,
	failedUnderlinesArr,
	setFailedUnderlinesArr,
}: AnalysisProps) {
	const [currentHovered, setCurrentHovered] = useState<number>();
	const [currentTab, setCurrentTab] = useState<string>('bias');

	// highlights, scroll to claim on sidepanel
	useEffect(() => {
		const highlightHoveredText = (message: any) => {
			if (message.type === 'UNDERLINE_HOVER') {
				setCurrentHovered(message.idx);
				const element = document.querySelector(`[claim-idx*="${message.idx}"]`);
				if (element) {
					element.scrollIntoView({ behavior: 'smooth', block: 'center' });
				}
			}
		};

		browser.runtime.onMessage.addListener(highlightHoveredText);
		return () => browser.runtime.onMessage.removeListener(highlightHoveredText);
	}, []);

	// highlights sentence on webpage
	async function handleHighlight(idx: number | undefined) {
		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;

		if (idx !== undefined) {
			await browser.tabs.sendMessage(tab.id, {
				type: 'HIGHLIGHT_TEXT',
				valid: currentTab === 'bias' ? result?.bias_claims[idx].valid : result?.fact_check_claims[idx].valid,
				idx: idx,
				category: result?.bias_claims[idx].category,
			});
		} else {
			await browser.tabs.sendMessage(tab.id, {
				type: 'CLEAR_HIGHLIGHTS',
			});
		}
	}

	// resets analysis
	async function newAnalysis() {
		setText(undefined);
		setResult(undefined);
		await browser.storage.local.remove('selectedText');

		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		await browser.tabs.sendMessage(tab.id ?? 0, {
			type: 'CLEAR_UNDERLINES',
		});
	}

	// switch between bias and claims tabs
	async function switchTab(value: string) {
		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;

		await browser.tabs.sendMessage(tab.id, {
			type: 'CLEAR_UNDERLINES',
		});
		if (value === 'claims') {
			const failed = await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				sentences: result?.fact_check_claims,
			});
			setFailedUnderlinesArr(failed);
		} else {
			const failed = await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				sentences: result?.bias_claims,
			});
			setFailedUnderlinesArr(failed);
		}
	}

	return (
		<>
			<Button variant="outline" className="w-full" onClick={newAnalysis}>
				New Analysis
			</Button>
			<Tabs
				defaultValue="bias"
				className="w-full gap-4"
				onValueChange={(e) => {
					setCurrentTab(e);
					switchTab(e);
				}}
			>
				<TabsList className="w-full h-fit flex justify-around p-2 rounded-full">
					<TabsTrigger
						value="bias"
						className="data-[state=active]:bg-white w-full data-[state=active]:shadow-sm py-1 rounded-full text-sm text-gray-950 font-semibold"
					>
						Bias
					</TabsTrigger>
					<TabsTrigger
						value="claims"
						className="data-[state=active]:bg-white w-full data-[state=active]:shadow-sm py-1 rounded-full text-sm text-gray-950 font-semibold"
					>
						Claims
					</TabsTrigger>
				</TabsList>
				<TabsContent value="bias" className="flex flex-col gap-4">
					<BiasTab
						key={currentTab}
						result={result}
						currentHovered={currentHovered}
						handleHighlight={handleHighlight}
						failedUnderlinesArr={failedUnderlinesArr}
					/>
				</TabsContent>
				<TabsContent value="claims" className="flex flex-col gap-4">
					<ClaimTab
						key={currentTab}
						result={result}
						currentHovered={currentHovered}
						handleHighlight={handleHighlight}
						failedUnderlinesArr={failedUnderlinesArr}
					/>
				</TabsContent>
			</Tabs>
		</>
	);
}
