import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { PieChart, Pie, Cell, Label } from 'recharts';
import { AnalysisResult, TokenAttribution } from '@/types';
import { SearchXIcon, ChevronDown, ChevronUp, SparklesIcon, Loader2Icon } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';
import { fetchExplain } from './fetchAPI';

type BiasTabProps = {
	result?: AnalysisResult;
	currentHovered?: number;
	handleHighlight: (index: number | undefined) => void;
	failedUnderlinesArr: number[];
};

function TokenChip({ token, score, category }: { token: string; score: number; category: string }) {
	const opacity = 0.15 + score * 0.65;
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
			<span className="text-gray-400 text-[10px]">{Math.round(score * 100)}%</span>
		</span>
	);
}

function ExplainabilitySection({
	tokens,
	category,
}: {
	tokens: TokenAttribution[];
	category: string;
}) {
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
	const [tokenCache, setTokenCache] = useState<Record<number, TokenAttribution[]>>({});
	const [loadingTokens, setLoadingTokens] = useState<Set<number>>(new Set());

	const toggleExpand = async (idx: number, e: React.MouseEvent) => {
		e.stopPropagation();

		const isExpanded = expandedCards.has(idx);
		setExpandedCards((prev) => {
			const next = new Set(prev);
			if (isExpanded) {
				next.delete(idx);
			} else {
				next.add(idx);
			}
			return next;
		});

		// fetch tokens on first expand if not cached
		if (!isExpanded && !tokenCache[idx] && result?.bias_claims[idx]) {
			setLoadingTokens((prev) => new Set(prev).add(idx));
			const tokens = await fetchExplain(result.bias_claims[idx].text);
			setTokenCache((prev) => ({ ...prev, [idx]: tokens }));
			setLoadingTokens((prev) => {
				const next = new Set(prev);
				next.delete(idx);
				return next;
			});
		}
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
	console.log(result?.bias_claims);

	return (
		<>
			<Card className="gap-0 rounded-4xl shadow-none py-5">
				<CardHeader className="flex gap-2 items-center justify-center">
					<p className="text-xl">Summary</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					{result?.overall_bias === 'Not Political' ? (
						<div className="flex flex-col items-center text-center gap-2 py-4">
							<p className="text-lg font-semibold">Not Political</p>
							<p className="text-sm text-gray-500">
								This text doesn't appear to contain political content.
							</p>
						</div>
					) : (
						<>
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
											(chartData[0].value /
												(chartData[0].value + chartData[1].value + chartData[2].value)) *
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
											(chartData[2].value /
												(chartData[0].value + chartData[1].value + chartData[2].value)) *
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
											(chartData[1].value /
												(chartData[0].value + chartData[1].value + chartData[2].value)) *
												100,
										)}
										%
									</p>
								</div>
							</div>
						</div>
					)}
						</>
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
									className={`flex flex-col p-0! gap-0 items-center hover:cursor-pointer border border-gray-200 rounded-xl ${
										claim.category.includes('Left')
											? 'hover:border-blue-400!'
											: claim.category.includes('Right')
												? 'hover:border-red-400!'
												: claim.category === 'Uncertain'
													? 'hover:border-gray-400!'
													: 'hover:border-purple-400!'
									}
										${
											currentHovered !== undefined &&
											idx === currentHovered &&
											(claim.category.includes('Left')
												? 'border-blue-400!'
												: claim.category.includes('Right')
													? 'border-red-400!'
													: claim.category === 'Uncertain'
														? 'border-gray-400!'
														: 'border-purple-400!')
										}`}
									claim-idx={idx}
									onMouseEnter={() => handleHighlight(idx)}
									onMouseLeave={() => handleHighlight(undefined)}
								>
									<CardHeader className="flex items-center w-full gap-2 border-b p-3!">
										{claim.category.includes('Left') ? (
											<div className="bg-blue-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												L
											</div>
										) : claim.category.includes('Right') ? (
											<div className="bg-red-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												R
											</div>
										) : claim.category === 'Uncertain' ? (
											<div className="bg-gray-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												?
											</div>
										) : (
											<div className="bg-purple-400 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
												C
											</div>
										)}
										<h3 className="text-base">{claim.category}</h3>
										<span className="ml-auto text-xs text-gray-400 font-medium">
											{Math.round(claim.confidence * 100)}%
										</span>

										{failedUnderlinesArr.includes(idx) && (
											<SearchXIcon className="ml-1" color="black" />
										)}
									</CardHeader>
									<CardContent className="p-3! w-full">
										<p className="text-sm line-clamp-6 text-gray-600">{claim.text}</p>

										{claim.confidence >= 0.70 && (
											<div className="mt-2">
												<button
													onClick={(e) => toggleExpand(idx, e)}
													className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition-colors"
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
													<>
														{loadingTokens.has(idx) ? (
															<div className="flex items-center gap-1.5 pt-2 text-xs text-gray-400">
																<Loader2Icon className="w-3 h-3 animate-spin" />
																<span>Analyzing...</span>
															</div>
														) : tokenCache[idx] && tokenCache[idx].length > 0 ? (
															<ExplainabilitySection
																tokens={tokenCache[idx]}
																category={claim.category}
															/>
														) : tokenCache[idx] && tokenCache[idx].length === 0 ? (
															<p className="text-xs text-gray-400 pt-1">
																No significant tokens found
															</p>
														) : null}
													</>
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
