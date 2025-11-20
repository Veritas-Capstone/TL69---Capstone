import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList } from '@/components/ui/tabs';
import { TabsTrigger } from '@radix-ui/react-tabs';
import { TextIcon } from 'lucide-react';
import BiasTab from './BiasTab';
import ClaimTab from './ClaimTab';

interface AnalysisResult {
	checks: number;
	issues: number;
	overall_bias: string;
	overall_probabilities: {
		Left: number;
		Center: number;
		Right: number;
	};
	bias_claims: { text: string; category: string; description: string; valid: boolean }[];
	fact_check_claims: { text: string; category: string; description: string; valid: boolean }[];
}

export default function AnalysisPage({
	text,
	setText,
	result,
	setResult,
}: {
	text: string | undefined;
	setText: React.Dispatch<React.SetStateAction<string | undefined>>;
	result: AnalysisResult | undefined;
	setResult: React.Dispatch<React.SetStateAction<AnalysisResult | undefined>>;
}) {
	const [currentHovered, setCurrentHovered] = useState<string>();
	const [currentTab, setCurrentTab] = useState<string>('bias');

	// highlights, scroll to claim on sidepanel
	useEffect(() => {
		const highlightHoveredText = (message: any) => {
			if (message.type === 'UNDERLINE_HOVER') {
				setCurrentHovered(message.text);
				console.log(message.text);
				const element = document.querySelector(`[claim-text*="${CSS.escape(message.text)}"]`);
				if (element) {
					element.scrollIntoView({ behavior: 'smooth', block: 'center' });
				}
			}
		};

		browser.runtime.onMessage.addListener(highlightHoveredText);
		return () => browser.runtime.onMessage.removeListener(highlightHoveredText);
	}, []);

	// highlights sentence on webpage
	async function handleHighlight(text: string) {
		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;

		if (text) {
			await browser.tabs.sendMessage(tab.id, {
				type: 'CLEAR_UNDERLINES',
			});
			await browser.tabs.sendMessage(tab.id, {
				type: 'HIGHLIGHT_TEXT',
				target: text,
			});
		} else {
			await browser.tabs.sendMessage(tab.id, {
				type: 'CLEAR_HIGHLIGHTS',
			});
			await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				valid: false,
				targets:
					currentTab === 'bias'
						? result?.bias_claims.filter((x) => !x.valid).map((x) => x.text)
						: result?.fact_check_claims.filter((x) => x.label !== 'SUPPORTED').map((x) => x.claim),
			});
			await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				valid: true,
				targets:
					currentTab === 'bias'
						? result?.bias_claims.filter((x) => x.valid).map((x) => x.text)
						: result?.fact_check_claims.filter((x) => x.label === 'SUPPORTED').map((x) => x.claim),
			});
		}
	}

	// resets analysis
	async function newAnalysis() {
		setText(undefined);
		setResult(undefined);
		await browser.storage.local.remove('storedResult');
		await browser.storage.local.remove('selectedText');

		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		await browser.tabs.sendMessage(tab.id ?? 0, {
			type: 'CLEAR_UNDERLINES',
		});
	}

	async function switchTab(value: string) {
		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;

		if (value === 'claims') {
			await browser.tabs.sendMessage(tab.id, {
				type: 'CLEAR_UNDERLINES',
			});
			await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				valid: false,
				targets: result?.fact_check_claims.filter((x) => !x.label !== 'SUPPORTED').map((x) => x.claim),
			});
			await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				valid: true,
				targets: result?.fact_check_claims.filter((x) => x.label === 'SUPPORTED').map((x) => x.claim),
			});
		} else {
			await browser.tabs.sendMessage(tab.id, {
				type: 'CLEAR_UNDERLINES',
			});
			await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				valid: false,
				targets: result?.bias_claims.filter((x) => !x.valid).map((x) => x.text),
			});
			await browser.tabs.sendMessage(tab.id ?? 0, {
				type: 'UNDERLINE_SELECTION',
				valid: true,
				targets: result?.bias_claims.filter((x) => x.valid).map((x) => x.text),
			});
		}
	}

	return (
		<CardContent className="w-full flex flex-col gap-4">
			<div className="flex items-center justify-between">
				<h1 className="font-semibold text-sm">Analysis Results</h1>
				<Button variant="link" className="p-0" onClick={newAnalysis}>
					New Analysis
				</Button>
			</div>
			<Card className="gap-2 h-[125px] overflow-y-auto py-3">
				<CardHeader className="flex gap-2 items-center">
					<TextIcon size={20} />
					<p className="font-semibold text-base">Inputted Text</p>
				</CardHeader>
				<CardContent className="flex flex-col">
					<div className="bg-gray-50 flex rounded-b-sm items-center">
						<p className="text-xs/relaxed text-gray-400">{text}</p>
					</div>
				</CardContent>
			</Card>

			<Tabs
				defaultValue="bias"
				className="max-w-xs w-full gap-4"
				onValueChange={(e) => {
					setCurrentTab(e);
					switchTab(e);
				}}
			>
				<TabsList className="w-full h-fit flex justify-around p-2">
					<TabsTrigger
						value="bias"
						className="data-[state=active]:bg-white w-full data-[state=active]:shadow-sm py-1 rounded-md text-sm text-gray-950 font-semibold"
					>
						Bias
					</TabsTrigger>
					<TabsTrigger
						value="claims"
						className="data-[state=active]:bg-white w-full data-[state=active]:shadow-sm py-1 rounded-md text-sm text-gray-950 font-semibold"
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
					/>
				</TabsContent>
				<TabsContent value="claims">
					<ClaimTab
						key={currentTab}
						result={result}
						currentHovered={currentHovered}
						handleHighlight={handleHighlight}
					/>
				</TabsContent>
			</Tabs>
		</CardContent>
	);
}
