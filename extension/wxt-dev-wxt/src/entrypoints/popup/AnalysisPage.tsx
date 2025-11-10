import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import {
	CheckCircleIcon,
	CircleXIcon,
	ShieldIcon,
	TextIcon,
	TrendingUpDownIcon,
	TriangleAlertIcon,
} from 'lucide-react';

export default function AnalysisPage({
	text,
	setText,
	result,
	setResult,
}: {
	text: string | undefined;
	setText: React.Dispatch<React.SetStateAction<string | undefined>>;
	result:
		| {
				checks: number;
				issues: number;
				claims: { text: string; category: string; description: string; valid: boolean }[];
		  }
		| undefined;
	setResult: React.Dispatch<
		React.SetStateAction<
			| {
					checks: number;
					issues: number;
					claims: { text: string; category: string; description: string; valid: boolean }[];
			  }
			| undefined
		>
	>;
}) {
	const [clicked, setClicked] = useState<string>();

	function normalizeSpaces(str: string) {
		return str
			.replace(/\s+/g, ' ')
			.replace(/\u00A0/g, ' ')
			.trim();
	}

	// when underlined text clicked, get the text and highlight corresponding claim
	useEffect(() => {
		async function getClicked() {
			const { clickedText } = await browser.storage.local.get('clickedText');
			await browser.storage.local.remove('clickedText');
			setClicked(clickedText);
		}

		getClicked();
	}, []);

	// highlight specific text on webpage
	async function handleHighlight(text: string) {
		if (clicked === text) {
			setClicked(undefined);
		}
		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;

		await browser.tabs.sendMessage(tab.id, {
			type: 'HIGHLIGHT_TEXT',
			target: text,
		});
	}

	// clears above highlights
	async function clearHighlights() {
		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;

		await browser.tabs.sendMessage(tab.id, {
			type: 'CLEAR_HIGHLIGHTS',
		});
	}

	async function newAnalysis() {
		setText(undefined);
		setResult(undefined);
		await browser.storage.local.remove('result');
		await browser.storage.local.remove('selectedText');

		const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
		if (!tab?.id) return;
		await browser.tabs.sendMessage(tab.id, {
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
			<div className="flex items-center justify-between gap-2">
				<Card className="flex flex-col gap-0 items-center justify-center w-full py-3">
					<CheckCircleIcon size="16" className="mb-2 text-green-400" />
					<p>{result?.checks}</p>
					<p>Checks</p>
				</Card>
				<Card className="flex flex-col gap-0 items-center justify-center w-full py-3">
					<CircleXIcon size="16" className="mb-2 text-red-400" />
					<p>{result?.issues}</p>
					<p>Issues</p>
				</Card>
			</div>
			<Card>
				<CardHeader className="flex gap-2 items-center">
					<TrendingUpDownIcon size={20} />
					<p className="font-semibold text-base">Bias Analysis</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					<div className="flex justify-between">
						<Badge className="bg-gray-400">Left wing</Badge>
						<p className="text-xs text-gray-500">70%</p>
					</div>
					<Progress value={70} className="[&>*]:bg-green-400" />
				</CardContent>
			</Card>
			<Card>
				<CardHeader className="flex gap-2 items-center">
					<ShieldIcon size={20} />
					<p className="font-semibold text-base">Fact Checks</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					{result?.claims.map((claim) => (
						<div
							className={`bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4 hover:cursor-pointer border-2 border-white ${
								clicked === claim.text && 'border-yellow-200'
							}`}
							onMouseEnter={() => handleHighlight(claim.text)}
							onMouseLeave={clearHighlights}
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
		</CardContent>
	);
}
