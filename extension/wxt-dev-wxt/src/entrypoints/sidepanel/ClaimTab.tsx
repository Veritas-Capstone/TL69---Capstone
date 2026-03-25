import { Card, CardHeader, CardContent } from '@/components/ui/card';
import {
	CheckCheckIcon,
	CheckIcon,
	ChevronDown,
	ChevronUp,
	CircleQuestionMarkIcon,
	SearchXIcon,
	SparklesIcon,
	XIcon,
} from 'lucide-react';
import { AnalysisResult } from '@/types';
import { PieChart, Pie, Cell, Label } from 'recharts';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';

type ClaimTabProps = {
	result?: AnalysisResult;
	currentHovered?: number;
	handleHighlight: (index: number | undefined) => void;
	failedUnderlinesArr: number[];
};

function TokenChip({ evidence }: { evidence: string }) {
	return (
		<Card className="p-2">
			<CardContent className="text-xs px-2 font-light">{evidence}</CardContent>
		</Card>
	);
}

function ExplainabilitySection({ evidence }: { evidence: string[] }) {
	if (!evidence || evidence.length === 0) return null;

	return (
		<div className="flex flex-wrap gap-1.5 pt-1">
			{evidence.map((e, i) => (
				<TokenChip key={`${e}-${i}`} evidence={e} />
			))}
		</div>
	);
}

export default function ClaimTab({
	result,
	currentHovered,
	handleHighlight,
	failedUnderlinesArr,
}: ClaimTabProps) {
	const [expandedCards, setExpandedCards] = useState<Set<number>>(new Set());

	const toggleExpand = (idx: number, e: React.MouseEvent) => {
		e.stopPropagation();
		setExpandedCards((prev) => {
			const next = new Set(prev);
			if (next.has(idx)) {
				next.delete(idx);
			} else {
				next.add(idx);
			}
			return next;
		});
	};

	const chartData = [
		{
			name: 'Supported',
			value: result?.fact_check_claims.filter((x) => x.label === 'SUPPORTED').length ?? 0,
		},
		{
			name: 'Refuted',
			value: result?.fact_check_claims.filter((x) => x.label === 'REFUTED').length ?? 0,
		},
		{
			name: 'Not enough info',
			value: result?.fact_check_claims.filter((x) => x.label === 'NOT ENOUGH INFO').length ?? 0,
		},
	];

	if (result?.fact_check_claims.length === 0) {
		return (
			<Card className="gap-0 rounded-4xl shadow-none py-5">
				<CardHeader className="flex gap-2 items-center justify-center">
					<p className="text-xl">No Claims Detected</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2 text-sm">
					This could be because the selected text is not making any fact-checkable claims and cannot be
					analyzed by our AI model.
				</CardContent>
			</Card>
		);
	}

	console.log(result?.fact_check_claims);

	return (
		<>
			<Card className="gap-0 rounded-4xl shadow-none py-5">
				<CardHeader className="flex gap-2 items-center justify-center">
					<p className="text-xl">Summary</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					<PieChart className="w-[70%] max-w-[172px] min-h-[172px] m-auto -my-2" responsive>
						<Pie
							data={chartData}
							dataKey="value"
							nameKey="name"
							fill="#8884d8"
							isAnimationActive={true}
							innerRadius={55}
						>
							<Label value={`${result?.fact_check_claims.length} Checks`} position={'center'} />
							{chartData.map((entry) => (
								<Cell
									fill={
										entry.name === 'Supported' ? '#4ade80' : entry.name === 'Refuted' ? '#f87171' : '#9ca3af'
									}
								/>
							))}
						</Pie>
					</PieChart>
					{result && (
						<div className="text-sm ml-auto mr-auto text-gray-500 w-full flex justify-center">
							<div className="flex justify-between gap-2 flex-col w-full text-base">
								<div className="flex justify-between w-full">
									<p>Supported</p>
									<Separator className="flex-[0.85] mt-3" />
									<p>
										{Math.round(
											(chartData[0].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
												100,
										)}
										%
									</p>
								</div>
								<div className="flex justify-between w-full">
									<p>Refuted</p>
									<Separator className="flex-[0.85] mt-3" />
									<p>
										{Math.round(
											(chartData[2].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
												100,
										)}
										%
									</p>
								</div>
								<div className="flex justify-between w-full">
									<p>Not Enough Info</p>
									<Separator className="flex-[0.85] mt-3" />
									<p>
										{Math.round(
											(chartData[1].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
												100,
										)}
										%
									</p>
								</div>
							</div>
						</div>
					)}
				</CardContent>
			</Card>
			<Card className="rounded-4xl shadow-none gap-4 py-5">
				<CardHeader className="flex gap-2 items-center justify-center">
					<p className="text-xl text-center">Sentence-Level Claims</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-4 pl-4 pr-2 max-h-[300px] overflow-auto">
					{result?.fact_check_claims.map((claim, idx) => (
						<Tooltip>
							<TooltipTrigger asChild>
								<Card
									key={`fact-${idx}`}
									className={`flex flex-col p-0! gap-0 items-center border border-gray-200 rounded-xl ${
										claim.label === 'SUPPORTED'
											? 'hover:border-green-400!'
											: claim.label === 'REFUTED'
												? 'hover:border-red-400!'
												: 'hover:border-gray-400!'
									}
											${
												currentHovered !== undefined &&
												idx === currentHovered &&
												(claim.label === 'SUPPORTED'
													? 'border-green-400!'
													: claim.label === 'REFUTED'
														? 'border-red-400!'
														: 'border-gray-400!')
											}`}
									claim-idx={idx}
									onMouseEnter={() => handleHighlight(idx)}
									onMouseLeave={() => handleHighlight(undefined)}
								>
									<CardHeader className="flex items-center w-full gap-2 border-b p-3!">
										{claim.label === 'SUPPORTED' ? (
											<div className="bg-green-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												<CheckIcon />
											</div>
										) : claim.label === 'REFUTED' ? (
											<div className="bg-red-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												<XIcon />
											</div>
										) : (
											<div className="bg-gray-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												?
											</div>
										)}
										<h3 className="text-base">{claim.label}</h3>

										{failedUnderlinesArr.includes(idx) && <SearchXIcon className="ml-auto" color="black" />}
									</CardHeader>
									<CardContent className="p-3! w-full">
										<p className="text-sm line-clamp-6 text-gray-600">{claim.claim}</p>

										{/* Explainability toggle */}
										{claim.evidence && claim.evidence.length > 0 && (
											<div>
												<button
													onClick={(e) => toggleExpand(idx, e)}
													className="flex w-full items-center justify-between gap-1 text-xs mt-1 mb-0.5 text-gray-400 hover:text-gray-600 transition-colors"
												>
													<span className="flex items-center gap-1">
														<SparklesIcon className="w-3 h-3" />
														<p>Evidence</p>
													</span>
													{expandedCards.has(idx) ? (
														<ChevronUp className="w-3 h-3" />
													) : (
														<ChevronDown className="w-3 h-3" />
													)}
												</button>

												{/* replace with actual evidence */}
												{expandedCards.has(idx) && <ExplainabilitySection evidence={claim.evidence} />}
											</div>
										)}
									</CardContent>
								</Card>
							</TooltipTrigger>
							{failedUnderlinesArr.includes(idx) && (
								<TooltipContent className="font-semibold text-center w-fit">
									Unable to locate on page
								</TooltipContent>
							)}
						</Tooltip>
					))}
				</CardContent>
			</Card>
		</>
	);
}
