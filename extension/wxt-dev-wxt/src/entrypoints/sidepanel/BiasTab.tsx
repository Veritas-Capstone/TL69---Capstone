import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { CheckCircleIcon, CircleXIcon, TrendingUpDownIcon, TriangleAlertIcon } from 'lucide-react';

interface AnalysisResult {
	checks: number;
	issues: number;
	overall_bias: string;
	overall_probabilities: {
		Left: number;
		Center: number;
		Right: number;
	};
	bias_claims: { text: string; category: string; description: string; valid: boolean }[];
	fact_check_claims: { text: string; category: string; description: string; valid: boolean }[];
}

export default function BiasTab({
	result,
	currentHovered,
	handleHighlight,
}: {
	result: AnalysisResult | undefined;
	currentHovered: string | undefined;
	handleHighlight: Function;
}) {
	const getBiasDisplay = () => {
		if (!result) return { label: 'Center', percentage: 50, color: 'bg-gray-400' };

		const probs = result.overall_probabilities;
		const maxProb = Math.max(probs.Left, probs.Center, probs.Right);
		const percentage = Math.round(maxProb * 100);

		if (result.overall_bias === 'Left') {
			return {
				label: 'Left wing',
				percentage,
				color: 'bg-blue-400',
			};
		} else if (result.overall_bias === 'Right') {
			return {
				label: 'Right wing',
				percentage,
				color: 'bg-red-400',
			};
		} else {
			return {
				label: 'Centrist',
				percentage,
				color: 'bg-gray-400',
			};
		}
	};

	const biasDisplay = getBiasDisplay();

	return (
		<>
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
						<Badge className={biasDisplay.color}>{biasDisplay.label}</Badge>
						<p className="text-xs text-gray-500">{biasDisplay.percentage}%</p>
					</div>
					<Progress value={biasDisplay.percentage} className={`[&>*]:${biasDisplay.color}`} />
					{result && (
						<div className="text-xs text-gray-500 mt-2">
							<div className="flex justify-between">
								<span>Left: {Math.round(result.overall_probabilities.Left * 100)}%</span>
								<span>Center: {Math.round(result.overall_probabilities.Center * 100)}%</span>
								<span>Right: {Math.round(result.overall_probabilities.Right * 100)}%</span>
							</div>
						</div>
					)}
				</CardContent>
			</Card>
			<Card>
				<CardHeader className="flex gap-2 items-center">
					<TrendingUpDownIcon size={20} />
					<p className="font-semibold text-base">Sentence-Level Bias</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					{result?.bias_claims.map((claim, idx) => (
						<div
							key={`bias-${idx}`}
							className={`bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4 hover:cursor-pointer border-2 border-white ${
								currentHovered && claim.text.includes(currentHovered) && 'border-yellow-200'
							}`}
							claim-text={claim.text}
							onMouseEnter={() => handleHighlight(claim.text)}
							onMouseLeave={() => handleHighlight('')}
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
		</>
	);
}
