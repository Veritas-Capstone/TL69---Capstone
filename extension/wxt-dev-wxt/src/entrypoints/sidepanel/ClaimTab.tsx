import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { CheckCheckIcon, CheckIcon, CircleQuestionMarkIcon, SearchXIcon, XIcon } from 'lucide-react';
import { AnalysisResult } from '@/types';
import { PieChart, Pie, Cell, Label } from 'recharts';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

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
	console.log(result?.fact_check_claims);
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
						<div className="text-sm ml-auto mr-auto text-gray-500">
							<div className="flex justify-between gap-4">
								<span>
									Supported:{' '}
									{Math.round(
										(chartData[0].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
											100,
									)}
									%
								</span>
								<span>
									Refuted:{' '}
									{Math.round(
										(chartData[1].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
											100,
									)}
									%
								</span>
								<span>
									Not enough info:{' '}
									{Math.round(
										(chartData[2].value / (chartData[0].value + chartData[1].value + chartData[2].value)) *
											100,
									)}
									%
								</span>
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
