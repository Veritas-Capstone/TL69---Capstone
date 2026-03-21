import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { PieChart, Pie, Cell, Label } from 'recharts';
import UserAuth from './UserAuth';
import { Stats } from '@/types';

export default function Profile() {
	const [userData, setUserData] = useState<string | undefined>(localStorage.getItem('username') ?? undefined);
	const [stats, setStats] = useState<Stats | undefined>();

	useEffect(() => {
		async function updateStats() {
			const response = await fetch(`http://localhost:8080/get-stats`, {
				headers: { 'Content-Type': 'application/json' },
				method: 'POST',
				body: JSON.stringify({ username: localStorage.getItem('username') }),
			});
			const data = await response.json();
			setStats(data);
		}

		if (localStorage.getItem('username')) {
			updateStats();
		}
	}, []);

	function getTotalBias() {
		return (stats?.leftBiasNum ?? 0) + (stats?.rightBiasNum ?? 0) + (stats?.centerBiasNum ?? 0);
	}

	function getTotalClaims() {
		return (stats?.supportedClaimNum ?? 0) + (stats?.refutedClaimNum ?? 0) + (stats?.noInfoClaimNum ?? 0);
	}

	function logout() {
		localStorage.clear();
		setUserData(undefined);
	}

	return (
		<div className="flex flex-col gap-4">
			{!userData ? (
				<UserAuth setUserData={setUserData} setStats={setStats} />
			) : (
				<>
					<div>
						<h1 className="text-2xl font-semibold">Welcome {userData}</h1>
						<p className="text-sm">See the full breakdown of the content you consume online.</p>
					</div>
					<div className="flex flex-col h-full">
						<div className="flex flex-col gap-4">
							<Card className="gap-0 rounded-4xl">
								<CardHeader>
									<h2 className="text-center text-lg">Bias Detection Stats</h2>
								</CardHeader>
								<CardContent>
									<PieChart className="w-[70%] max-w-[172px] min-h-[172px] m-auto -my-2" responsive>
										<Pie
											data={[
												{
													name: 'Left',
													value: Math.max(stats?.leftBiasNum ?? 0, 1),
												},
												{
													name: 'Right',
													value: Math.max(stats?.rightBiasNum ?? 0, 1),
												},
												{
													name: 'Center',
													value: Math.max(stats?.centerBiasNum ?? 0, 1),
												},
											]}
											dataKey="value"
											nameKey="name"
											fill="#8884d8"
											isAnimationActive={true}
											innerRadius={55}
										>
											<Label value={`${getTotalBias()} Checks Total`} position={'center'} />
											{[
												{
													name: 'Left',
													value: stats?.leftBiasNum ?? 0,
												},
												{
													name: 'Right',
													value: stats?.rightBiasNum ?? 0,
												},
												{
													name: 'Center',
													value: stats?.centerBiasNum ?? 0,
												},
											].map((entry) => (
												<Cell
													fill={
														entry.name === 'Left' ? '#60a5fa' : entry.name === 'Right' ? '#f87171' : '#c084fc'
													}
												/>
											))}
										</Pie>
									</PieChart>
									<div className="text-sm ml-auto mr-auto text-gray-500">
										<div className="flex justify-between gap-4">
											<span>
												Left: {Math.round(((stats?.leftBiasNum ?? 0) / Math.max(getTotalBias(), 1)) * 100)}%
											</span>
											<span>
												Center:{' '}
												{Math.round(((stats?.centerBiasNum ?? 0) / Math.max(getTotalBias(), 1)) * 100)}%
											</span>
											<span>
												Right: {Math.round(((stats?.rightBiasNum ?? 0) / Math.max(getTotalBias(), 1)) * 100)}%
											</span>
										</div>
									</div>
								</CardContent>
							</Card>
							<Card className="gap-0 rounded-4xl">
								<CardHeader>
									<h2 className="text-center text-lg">Claim Verification Stats</h2>
								</CardHeader>
								<CardContent>
									<PieChart className="w-[70%] max-w-[172px] min-h-[172px] m-auto -my-2" responsive>
										<Pie
											data={[
												{
													name: 'Supported',
													value: Math.max(stats?.supportedClaimNum ?? 0, 1),
												},
												{
													name: 'Refuted',
													value: Math.max(stats?.refutedClaimNum ?? 0, 1),
												},
												{
													name: 'NoInfo',
													value: Math.max(stats?.noInfoClaimNum ?? 0, 1),
												},
											]}
											dataKey="value"
											nameKey="name"
											fill="#8884d8"
											isAnimationActive={true}
											innerRadius={55}
										>
											<Label value={`${getTotalClaims()} Checks Total`} position={'center'} />
											{[
												{
													name: 'Supported',
													value: stats?.supportedClaimNum ?? 0,
												},
												{
													name: 'Refuted',
													value: stats?.refutedClaimNum ?? 0,
												},
												{
													name: 'NoInfo',
													value: stats?.noInfoClaimNum ?? 0,
												},
											].map((entry) => (
												<Cell
													fill={
														entry.name === 'Supported'
															? '#4ade80'
															: entry.name === 'Refuted'
																? '#f87171'
																: '#9ca3af'
													}
												/>
											))}
										</Pie>
									</PieChart>
									<div className="text-sm ml-auto mr-auto text-gray-500">
										<div className="flex justify-between gap-4">
											<span>
												Supported:{' '}
												{Math.round(((stats?.supportedClaimNum ?? 0) / Math.max(getTotalClaims(), 1)) * 100)}%
											</span>
											<span>
												Refuted:{' '}
												{Math.round(((stats?.refutedClaimNum ?? 0) / Math.max(getTotalClaims(), 1)) * 100)}%
											</span>
											<span>
												No Info:{' '}
												{Math.round(((stats?.noInfoClaimNum ?? 0) / Math.max(getTotalClaims(), 1)) * 100)}%
											</span>
										</div>
									</div>
								</CardContent>
							</Card>
						</div>

						<Button onClick={logout} className="mt-10 ml-auto">
							Logout
						</Button>
					</div>
				</>
			)}
		</div>
	);
}
