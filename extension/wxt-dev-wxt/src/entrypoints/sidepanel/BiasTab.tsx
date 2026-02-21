import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { PieChart, Pie, Cell, Label } from 'recharts';
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
					(x, idx) => x.category === 'Left-leaning' && !failedUnderlinesArr.includes(idx),
				).length ?? 0,
		},
		{
			name: 'Right',
			value:
				result?.bias_claims.filter(
					(x, idx) => x.category === 'Right-leaning' && !failedUnderlinesArr.includes(idx),
				).length ?? 0,
		},
		{
			name: 'Center',
			value:
				result?.bias_claims.filter(
					(x, idx) => x.category === 'Centrist' && !failedUnderlinesArr.includes(idx),
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
								<div className="text-sm ml-auto mr-auto text-gray-500">
									<div className="flex justify-between gap-4">
										<span>
											Left:{' '}
											{Math.round(
												(chartData[0].value /
													(chartData[0].value + chartData[1].value + chartData[2].value)) *
													100,
											)}
											%
										</span>
										<span>
											Center:{' '}
											{Math.round(
												(chartData[2].value /
													(chartData[0].value + chartData[1].value + chartData[2].value)) *
													100,
											)}
											%
										</span>
										<span>
											Right:{' '}
											{Math.round(
												(chartData[1].value /
													(chartData[0].value + chartData[1].value + chartData[2].value)) *
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
							{result?.bias_claims.map(
								(claim, idx) =>
									!failedUnderlinesArr.includes(idx) && (
										<Card
											key={`bias-${idx}`}
											className={`flex flex-col p-0! gap-0 items-center hover:cursor-pointer border border-gray-200 rounded-xl ${
												claim.category === 'Left-leaning'
													? 'hover:border-blue-500!'
													: claim.category === 'Right-leaning'
														? 'hover:border-red-500!'
														: 'hover:border-purple-500!'
											}
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
											<CardHeader className="flex items-center w-full gap-2 border-b p-3!">
												{claim.category === 'Left-leaning' ? (
													<div className="bg-blue-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
														L
													</div>
												) : claim.category === 'Right-leaning' ? (
													<div className="bg-red-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
														R
													</div>
												) : (
													<div className="bg-purple-500 text-white font-bold min-w-6 h-6 flex justify-center items-center text-lg mb-auto rounded-sm">
														C
													</div>
												)}
												<h3 className="text-base">{claim.category}</h3>
											</CardHeader>
											<CardContent className="p-3! w-full">
												<p className="text-sm line-clamp-6 text-gray-600">{claim.text}</p>
											</CardContent>
										</Card>
									),
							)}
						</CardContent>
					</Card>
				</>
			</>
		);
	}
}
