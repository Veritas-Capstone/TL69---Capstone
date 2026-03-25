import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { PieChart, Pie, Cell, Label } from 'recharts';
import { AnalysisResult, TokenAttribution } from '@/types';
import { SearchXIcon, ChevronDown, ChevronUp, SparklesIcon } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';

type BiasTabProps = {
	result?: AnalysisResult;
	currentHovered?: number;
	handleHighlight: (index: number | undefined) => void;
	failedUnderlinesArr: number[];
};

function TokenChip({ token, score, category }: { token: string; score: number; category: string }) {
	// Intensity based on attribution score (0-1)
	const opacity = 0.05 + score * 0.6;
	const bgColor =
		category === 'Left-leaning'
			? `rgba(96, 165, 250, ${opacity})`
			: category === 'Right-leaning'
				? `rgba(248, 113, 113, ${opacity})`
				: `rgba(192, 132, 252, ${opacity})`;

	const borderColor =
		category === 'Left-leaning'
			? 'rgba(96, 165, 250, 0.5)'
			: category === 'Right-leaning'
				? 'rgba(248, 113, 113, 0.5)'
				: 'rgba(192, 132, 252, 0.5)';

	return (
		<span
			className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
			style={{
				backgroundColor: bgColor,
				border: `1px solid ${borderColor}`,
			}}
		>
			{token}
			<span className="text-gray-500 text-[10px]">{Math.round(score * 100)}%</span>
		</span>
	);
}

function ExplainabilitySection({ tokens, category }: { tokens: TokenAttribution[]; category: string }) {
	if (!tokens || tokens.length === 0) return null;

	return (
		<div className="flex flex-wrap gap-1.5 pt-1">
			{tokens.map((t, i) => (
				<TokenChip key={`${t.token}-${i}`} token={t.token} score={t.score} category={category} />
			))}
		</div>
	);
}

export default function BiasTab({
	result,
	currentHovered,
	handleHighlight,
	failedUnderlinesArr,
}: BiasTabProps) {
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
			name: 'Left',
			value: result?.bias_claims.filter((x) => x.category === 'Left-leaning').length ?? 0,
		},
		{
			name: 'Right',
			value: result?.bias_claims.filter((x) => x.category === 'Right-leaning').length ?? 0,
		},
		{
			name: 'Center',
			value:
				result?.bias_claims.filter((x) => x.category === 'Centrist' || x.category === 'Neutral/Balanced')
					.length ?? 0,
		},
	];

	if (result?.bias_claims.length === 0) {
		return (
			<Card className="gap-0 rounded-4xl shadow-none py-5">
				<CardHeader className="flex gap-2 items-center justify-center">
					<p className="text-xl">No Bias Detected</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2 text-sm">
					This could be because the selected text is not biased and cannot be analyzed by our AI model.
				</CardContent>
			</Card>
		);
	}

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
							<Label value={`${result?.bias_claims.length} Checks`} position={'center'} />
							{chartData.map((entry, i) => (
								<Cell
									key={`cell-${i}`}
									fill={entry.name === 'Left' ? '#60a5fa' : entry.name === 'Right' ? '#f87171' : '#c084fc'}
								/>
							))}
						</Pie>
					</PieChart>
					{result && (
						<div className="text-sm ml-auto mr-auto text-gray-500 w-full flex justify-center">
							<div className="flex justify-between gap-2 flex-col w-full text-base">
								<div className="flex justify-between w-full">
									<p>Left</p>
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
									<p>Center</p>
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
									<p>Right</p>
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
					<p className="text-xl">Sentence-Level Bias</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-4 pl-4 pr-2 max-h-[300px] overflow-auto">
					{result?.bias_claims.map((claim, idx) => (
						<Tooltip key={`bias-${idx}`}>
							<TooltipTrigger asChild>
								<Card
									className={`flex flex-col p-0! gap-0 items-center border border-gray-200 rounded-xl ${
										claim.category === 'Left-leaning'
											? 'hover:border-blue-400!'
											: claim.category === 'Right-leaning'
												? 'hover:border-red-400!'
												: 'hover:border-purple-400!'
									}
										${
											currentHovered !== undefined &&
											idx === currentHovered &&
											(claim.category === 'Left-leaning'
												? 'border-blue-400!'
												: claim.category === 'Right-leaning'
													? 'border-red-400!'
													: 'border-purple-400!')
										}`}
									claim-idx={idx}
									onMouseEnter={() => handleHighlight(idx)}
									onMouseLeave={() => handleHighlight(undefined)}
								>
									<CardHeader className="flex items-center w-full gap-2 border-b p-3!">
										{claim.category === 'Left-leaning' ? (
											<div className="bg-blue-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												L
											</div>
										) : claim.category === 'Right-leaning' ? (
											<div className="bg-red-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												R
											</div>
										) : (
											<div className="bg-purple-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												C
											</div>
										)}
										<h3 className="text-base">{claim.category}</h3>

										{failedUnderlinesArr.includes(idx) && <SearchXIcon className="ml-auto" color="black" />}
									</CardHeader>
									<CardContent className="p-3! w-full">
										<p className="text-sm line-clamp-6 text-gray-600">{claim.text}</p>

										{/* Explainability toggle */}
										{claim.top_tokens && claim.top_tokens.length > 0 && (
											<div>
												<button
													onClick={(e) => toggleExpand(idx, e)}
													className="flex items-center gap-1 text-xs mt-1 mb-0.5 text-gray-400 hover:text-gray-600 transition-colors"
												>
													<SparklesIcon className="w-3 h-3" />
													<span>Words that influenced this score</span>
													{expandedCards.has(idx) ? (
														<ChevronUp className="w-3 h-3" />
													) : (
														<ChevronDown className="w-3 h-3" />
													)}
												</button>
												{expandedCards.has(idx) && (
													<ExplainabilitySection tokens={claim.top_tokens} category={claim.category} />
												)}
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
