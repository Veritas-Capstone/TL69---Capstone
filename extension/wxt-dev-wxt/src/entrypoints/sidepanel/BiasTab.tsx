import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { PieChart, Pie, Cell, Label } from 'recharts';
import { ChartColumnBigIcon, SearchCheckIcon } from 'lucide-react';
import { AnalysisResult } from '@/types';

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
			value:
				result?.bias_claims.filter(
					(x, idx) => x.category === 'Left-leaning' && !failedUnderlinesArr.includes(idx)
				).length ?? 0,
		},
		{
			name: 'Right',
			value:
				result?.bias_claims.filter(
					(x, idx) => x.category === 'Right-leaning' && !failedUnderlinesArr.includes(idx)
				).length ?? 0,
		},
		{
			name: 'Center',
			value:
				result?.bias_claims.filter(
					(x, idx) => x.category === 'Centrist' && !failedUnderlinesArr.includes(idx)
				).length ?? 0,
		},
	];

	if ((result?.bias_claims.length ?? 0) - failedUnderlinesArr.length < 1) {
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
				<>
					<Card className="gap-0">
						<CardHeader className="flex gap-2 items-center">
							<ChartColumnBigIcon size={20} />
							<p className="font-semibold text-base">Bias Summary</p>
						</CardHeader>
						<CardContent className="flex flex-col gap-2">
							<PieChart className="w-[70%] max-w-[150px] min-h-[150px] m-auto" responsive>
								<Pie
									data={chartData}
									dataKey="value"
									nameKey="name"
									fill="#8884d8"
									isAnimationActive={true}
									innerRadius={45}
								>
									<Label
										value={`${
											result?.bias_claims.filter((_, idx) => !failedUnderlinesArr.includes(idx)).length
										} Checks`}
										position={'center'}
									/>
									{chartData.map((entry) => (
										<Cell
											fill={
												entry.name === 'Left' ? '#3b82f6' : entry.name === 'Right' ? '#ef4444' : '#8b5cf6'
											}
										/>
									))}
								</Pie>
							</PieChart>
							{result && (
								<div className="text-xs max-w-[225px] ml-auto mr-auto text-gray-500">
									<div className="flex justify-between gap-4">
										<span>
											Left:{' '}
											{Math.round(
												(chartData[0].value /
													(chartData[0].value + chartData[1].value + chartData[2].value)) *
													100
											)}
											%
										</span>
										<span>
											Center:{' '}
											{Math.round(
												(chartData[2].value /
													(chartData[0].value + chartData[1].value + chartData[2].value)) *
													100
											)}
											%
										</span>
										<span>
											Right:{' '}
											{Math.round(
												(chartData[1].value /
													(chartData[0].value + chartData[1].value + chartData[2].value)) *
													100
											)}
											%
										</span>
									</div>
								</div>
							)}
						</CardContent>
					</Card>
					<Card>
						<CardHeader className="flex gap-2 items-center">
							<SearchCheckIcon size={20} />
							<p className="font-semibold text-base">Sentence-Level Bias</p>
						</CardHeader>
						<CardContent className="flex flex-col gap-2">
							{result?.bias_claims.map(
								(claim, idx) =>
									!failedUnderlinesArr.includes(idx) && (
										<div
											key={`bias-${idx}`}
											className={`bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4 hover:cursor-pointer border-2 border-white
											${
												currentHovered !== undefined &&
												idx === currentHovered &&
												(claim.category === 'Left-leaning'
													? 'border-blue-500!'
													: claim.category === 'Right-leaning'
													? 'border-red-500!'
													: 'border-purple-500!')
											}`}
											claim-idx={idx}
											onMouseEnter={() => handleHighlight(idx)}
											onMouseLeave={() => handleHighlight(undefined)}
										>
											{claim.category === 'Left-leaning' ? (
												<div className="bg-blue-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg">
													L
												</div>
											) : claim.category === 'Right-leaning' ? (
												<div className="bg-red-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg">
													R
												</div>
											) : (
												<div className="bg-purple-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg">
													C
												</div>
											)}
											<div className="flex flex-col">
												<h3 className="text-sm">{claim.category}</h3>
												<p className="text-xs text-gray-400">{claim.description}</p>
											</div>
										</div>
									)
							)}
						</CardContent>
					</Card>
				</>
			</>
		);
	}
}
