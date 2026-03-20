import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { ShieldIcon, CheckCircleIcon, TriangleAlertIcon, CircleXIcon } from 'lucide-react';
import { AnalysisResult } from '@/types';

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
				<div className="flex items-center justify-between gap-2">
					<Card className="flex flex-col gap-0 items-center justify-center w-full py-3">
						<CheckCircleIcon size="16" className="mb-2 text-green-400" />
						<p>{(result?.fact_check_claims.length ?? 0) - failedUnderlinesArr.length}</p>
						<p>Checks</p>
					</Card>
					<Card className="flex flex-col gap-0 items-center justify-center w-full py-3">
						<CircleXIcon size="16" className="mb-2 text-red-400" />
						<p>
							{
								result?.fact_check_claims.filter(
									(claim, idx) => !claim.valid && !failedUnderlinesArr.includes(idx),
								).length
							}
						</p>
						<p>Issues</p>
					</Card>
				</div>
				<Card>
					<CardHeader className="flex gap-2 items-center">
						<ShieldIcon size={20} />
						<p className="font-semibold text-base">Claim Verification</p>
					</CardHeader>
					<CardContent className="flex flex-col gap-2">
						{result?.fact_check_claims.map(
							(claim, idx) =>
								!failedUnderlinesArr.includes(idx) && (
									<div
										key={`fact-${idx}`}
										className={`bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4 hover:cursor-pointer border-2 border-white 
								${currentHovered !== undefined && idx === currentHovered && 'border-yellow-200'}`}
										claim-idx={idx}
										onMouseEnter={() => handleHighlight(idx)}
										onMouseLeave={() => handleHighlight(undefined)}
									>
										{claim.valid ? (
											<CheckCircleIcon className="min-w-6 h-6 text-green-400" />
										) : (
											<TriangleAlertIcon className="min-w-6 h-6 text-red-400" />
										)}
										<div className="flex flex-col">
											<h3 className="text-sm">{claim.label}</h3>
											{claim.evidence.map((ev) => (
												<p className="text-xs text-gray-400">- {ev}</p>
											))}
										</div>
									</div>
								),
						)}
					</CardContent>
				</Card>
			</>
		);
	}
}
