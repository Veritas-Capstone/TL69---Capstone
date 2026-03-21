import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { SearchXIcon } from 'lucide-react';
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

	if ((result?.fact_check_claims.length ?? 0) - failedUnderlinesArr.length < 1) {
		return (
			<Card>
				<CardHeader className="flex gap-2 items-center">
					<p className="font-semibold text-base">No Text Detected</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">Please try again</CardContent>
			</Card>
		);
	} else {
		return (
			<>
				<Card className="gap-0 rounded-4xl shadow-none">
					<CardHeader className="flex gap-2 items-center">
						<p className="font-semibold text-xl">Summary</p>
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
											entry.name === 'Supported'
												? '#3b82f6'
												: entry.name === 'Refuted'
													? '#ef4444'
													: '#6b7280'
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
				<Card className="rounded-4xl shadow-none">
					<CardHeader className="flex gap-2 items-center">
						<p className="font-semibold text-xl">Sentence-Level Bias</p>
					</CardHeader>
					<CardContent className="flex flex-col gap-4 px-4">
						{result?.fact_check_claims.map((claim, idx) => (
							<Tooltip>
								<TooltipTrigger asChild>
									<Card
										key={`fact-${idx}`}
										className={`flex flex-col p-0! gap-0 items-center hover:cursor-pointer border border-gray-200 rounded-xl ${
											claim.label === 'SUPPORTED'
												? 'hover:border-blue-500!'
												: claim.label === 'REFUTED'
													? 'hover:border-red-500!'
													: 'hover:border-gray-500!'
										}
											${
												currentHovered !== undefined &&
												idx === currentHovered &&
												(claim.label === 'SUPPORTED'
													? 'border-blue-500!'
													: claim.label === 'REFUTED'
														? 'border-red-500!'
														: 'border-gray-500!')
											}`}
										claim-idx={idx}
										onMouseEnter={() => handleHighlight(idx)}
										onMouseLeave={() => handleHighlight(undefined)}
									>
										<CardHeader className="flex items-center w-full gap-2 border-b p-3!">
											{claim.label === 'SUPPORTED' ? (
												<div className="bg-blue-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
													S
												</div>
											) : claim.label === 'REFUTED' ? (
												<div className="bg-red-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
													R
												</div>
											) : (
												<div className="bg-gray-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
													N
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
}
