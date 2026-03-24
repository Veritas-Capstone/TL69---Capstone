import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { PieChart, Pie, Cell, Label } from 'recharts';
import { AnalysisResult } from '@/types';
import { SearchXIcon } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Separator } from '@/components/ui/separator';

type BiasTabProps = {
	result?: AnalysisResult;
	currentHovered?: number;
	handleHighlight: (index: number | undefined) => void;
	failedUnderlinesArr: number[];
};

export default function BiasTab({
	result,
	currentHovered,
	handleHighlight,
	failedUnderlinesArr,
}: BiasTabProps) {
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
								{chartData.map((entry) => (
									<Cell
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
					</CardContent>
				</Card>
				<Card className="rounded-4xl shadow-none gap-4 py-5">
					<CardHeader className="flex gap-2 items-center justify-center">
						<p className="text-xl">Sentence-Level Bias</p>
					</CardHeader>
					<CardContent className="flex flex-col gap-4 pl-4 pr-2 max-h-[300px] overflow-auto">
						{result?.bias_claims.map((claim, idx) => (
							<Tooltip>
								<TooltipTrigger asChild>
									<Card
										key={`bias-${idx}`}
										className={`flex flex-col p-0! gap-0 items-center hover:cursor-pointer border border-gray-200 rounded-xl ${
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
		</>
	);
}
