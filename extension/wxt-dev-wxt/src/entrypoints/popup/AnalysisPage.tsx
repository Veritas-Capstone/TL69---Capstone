import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { CheckCircleIcon, ShieldIcon, TextIcon, TrendingUpDownIcon, TriangleAlertIcon } from 'lucide-react';

export default function AnalysisPage({
	text,
	setText,
}: {
	text: string;
	setText: React.Dispatch<React.SetStateAction<string | undefined>>;
}) {
	return (
		<CardContent className="w-full flex flex-col gap-4">
			<div className="flex items-center justify-between">
				<h1 className="font-semibold text-sm">Analysis Results</h1>
				<Button variant="link" className="p-0" onClick={() => setText(undefined)}>
					Back
				</Button>
			</div>
			<Card className="gap-2 max-h-[125px] overflow-y-auto py-3">
				<CardHeader className="flex gap-2 items-center">
					<TextIcon size={20} />
					<p className="font-semibold text-base">Inputted Text</p>
				</CardHeader>
				<CardContent className="flex flex-col">
					<div className="bg-gray-50 flex rounded-b-sm items-center">
						<p className="text-xs/relaxed text-gray-400">{text}</p>
					</div>
				</CardContent>
			</Card>
			<div className="flex items-center justify-between gap-2">
				<Card className="flex flex-col gap-0 items-center justify-center w-full py-3">
					<CheckCircleIcon size="16" className="mb-2 text-green-400" />
					<p>2</p>
					<p>Checks</p>
				</Card>
				<Card className="flex flex-col gap-0 items-center justify-center w-full py-3">
					<CheckCircleIcon size="16" className="mb-2 text-green-400" />
					<p>0</p>
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
						<Badge className="bg-green-400">Center</Badge>
						<p className="text-xs text-gray-500">100%</p>
					</div>
					<Progress value={100} className="[&>*]:bg-green-400" />
				</CardContent>
			</Card>
			<Card>
				<CardHeader className="flex gap-2 items-center">
					<ShieldIcon size={20} />
					<p className="font-semibold text-base">Fact Checks</p>
				</CardHeader>
				<CardContent className="flex flex-col gap-2">
					<div className="bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4">
						<TriangleAlertIcon size={16} />
						<div className="flex flex-col">
							<h3 className="text-sm">Statistical claim in article</h3>
							<p className="text-xs text-gray-400">Numbers appear rounded or outdated.</p>
						</div>
					</div>
					<div className="bg-gray-50 flex rounded-b-sm gap-4 items-center py-2 pl-4">
						<CheckCircleIcon size={16} />
						<div className="flex flex-col">
							<h3 className="text-sm">Policy impact statement</h3>
							<p className="text-xs text-gray-400">Supported by official data.</p>
						</div>
					</div>
				</CardContent>
			</Card>
		</CardContent>
	);
}
