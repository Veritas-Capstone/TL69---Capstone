import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList } from '@/components/ui/tabs';
import { TabsTrigger } from '@radix-ui/react-tabs';
import {
	CheckCircleIcon,
	CircleXIcon,
	ShieldIcon,
	TextIcon,
	TrendingUpDownIcon,
	TriangleAlertIcon,
} from 'lucide-react';
import BiasTab from './BiasTab';

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

	// highlights, scroll to claim on sidepanel
	useEffect(() => {
		const highlightHoveredText = (message: any) => {
			if (message.type === 'UNDERLINE_HOVER') {
				setCurrentHovered(message.text);
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
				type: 'HIGHLIGHT_TEXT',
				target: text,
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
		await browser.storage.local.remove('storedResult');
		await browser.storage.local.remove('selectedText');

		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		await browser.tabs.sendMessage(tab.id ?? 0, {
			type: 'CLEAR_UNDERLINES',
		});
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

			<Tabs defaultValue="bias" className="max-w-xs w-full gap-4">
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
					<BiasTab result={result} currentHovered={currentHovered} handleHighlight={handleHighlight} />
				</TabsContent>
				{/* Mocking claim verification */}
				<TabsContent value="claims">
					<Card>
						<CardHeader className="flex gap-2 items-center">
							<ShieldIcon size={20} />
							<p className="font-semibold text-base">Claim Verification</p>
						</CardHeader>
						<CardContent className="flex flex-col gap-2">
							<div className="bg-yellow-50 border border-yellow-200 rounded p-2 mb-2">
								<p className="text-xs text-yellow-800">This section shows mock data for now.</p>
							</div>
							{result?.fact_check_claims.map((claim, idx) => (
								<div
									key={`fact-${idx}`}
									className={`bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4 hover:cursor-pointer border-2 border-white ${
										currentHovered === claim.text && 'border-yellow-200'
									}`}
									claim-text={claim.text}
									onMouseEnter={() => handleHighlight(claim.text)}
									onMouseLeave={() => handleHighlight('')}
								>
									{claim.valid ? (
										<CheckCircleIcon className="w-12 h-12 text-green-400" />
									) : (
										<TriangleAlertIcon className="w-12 h-12 text-red-400" />
									)}
									<div className="flex flex-col">
										<h3 className="text-sm">{claim.category}</h3>
										<p className="text-xs text-gray-400">{claim.description}</p>
									</div>
								</div>
							))}
						</CardContent>
					</Card>
				</TabsContent>
			</Tabs>
		</CardContent>
	);
}
