import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { ShieldIcon, CheckCircleIcon, TriangleAlertIcon } from 'lucide-react';

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

export default function ClaimTab({
	result,
	currentHovered,
	handleHighlight,
}: {
	result: AnalysisResult | undefined;
	currentHovered: string | undefined;
	handleHighlight: Function;
}) {
	return (
		<>
			<Card>
				<CardHeader className="flex gap-2 items-center">
					<ShieldIcon size={20} />
					<p className="font-semibold text-base">Claim Verification</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					{result?.fact_check_claims.map((claim, idx) => (
						<div
							key={`fact-${idx}`}
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
