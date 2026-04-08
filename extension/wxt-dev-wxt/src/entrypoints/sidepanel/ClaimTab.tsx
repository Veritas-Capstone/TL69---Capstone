import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { CheckIcon, SearchXIcon, XIcon } from 'lucide-react';
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

export default function ClaimTab({
	result,
	currentHovered,
	handleHighlight,
	failedUnderlinesArr,
}: ClaimTabProps) {
	const formatSourceLabel = (sourceRef?: string) => {
		if (!sourceRef) {
			return 'Source unavailable';
		}

		try {
			const url = new URL(sourceRef);
			return url.hostname.replace(/^www\./, '');
		} catch {
			return sourceRef;
		}
	};

	const isHttpUrl = (value?: string) => Boolean(value && /^https?:\/\//i.test(value));

	const formatEvidenceSource = (source?: string) => {
		if (!source) {
			return 'Wikipedia';
		}

		if (source === 'local+web') {
			return 'Wikipedia + live web fallback';
		}

		if (source === 'local') {
			return 'Wikipedia';
		}

		return source.replace(/_/g, ' ');
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
									<p>Supported ({chartData[0].value})</p>
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
									<p>Refuted ({chartData[1].value})</p>
									<Separator className="flex-[0.85] mt-3" />
									<p>
										{Math.round(
											(chartData[1].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
												100,
										)}
										%
									</p>
								</div>
								<div className="flex justify-between w-full">
									<p>Not Enough Info ({chartData[2].value})</p>
									<Separator className="flex-[0.85] mt-3" />
									<p>
										{Math.round(
											(chartData[2].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
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
					<p className="text-xs text-gray-500 text-center px-3">
						Evidence is retrieved from Wikipedia (with optional web fallback) and linked sources are shown when available.
					</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-4 pl-4 pr-2 max-h-[300px] overflow-auto">
					{result?.fact_check_claims.map((claim, idx) => (
						<Tooltip key={`fact-${idx}`}>
							<TooltipTrigger asChild>
								<Card
									className={`flex flex-col p-0! gap-0 items-center hover:cursor-pointer border border-gray-200 rounded-xl ${
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
										<details className="mt-3 rounded-2xl border border-gray-200 bg-gray-50 px-3 py-3 group">
											<summary className="flex items-center justify-between gap-2 list-none cursor-pointer">
												<div>
													<p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-gray-500">Cited Evidence</p>
													<p className="text-[11px] text-gray-500 mt-1">Source: {formatEvidenceSource(claim.evidence_source)}</p>
												</div>
												<div className="text-right">
													<p className="text-[11px] text-gray-400">{claim.evidence?.length ?? 0} snippet(s)</p>
													<p className="text-[11px] text-blue-600 group-open:hidden">Show</p>
													<p className="text-[11px] text-blue-600 hidden group-open:block">Hide</p>
												</div>
											</summary>
											<div className="mt-3 space-y-2">
												{claim.evidence?.length ? (
													claim.evidence.map((snippet, evidenceIdx) => {
														const sourceRef = claim.evidence_links?.[evidenceIdx] ?? '';
														const sourceLabel = formatSourceLabel(sourceRef);
														const sourceHref = isHttpUrl(sourceRef) ? sourceRef : undefined;

														return (
															<div key={`fact-${idx}-evidence-${evidenceIdx}`} className="rounded-xl border border-gray-200 bg-white px-3 py-2 shadow-sm">
																<p className="text-sm leading-5 text-gray-700">{snippet}</p>
																<div className="mt-2 flex items-center justify-between gap-3 text-[11px] text-gray-500">
																	<span className="truncate">{sourceLabel}</span>
																	{sourceHref ? (
																		<a
																			href={sourceHref}
																			target="_blank"
																			rel="noreferrer"
																			className="shrink-0 font-medium text-blue-600 hover:underline"
																		>
																			Open source
																		</a>
																	) : null}
																</div>
															</div>
														);
													})
												) : (
													<p className="text-sm text-gray-400">No cited evidence returned for this claim.</p>
												)}
											</div>
										</details>
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
