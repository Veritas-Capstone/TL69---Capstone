import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { PieChart, Pie, Cell, Label } from 'recharts';
import UserAuth from './UserAuth';
import { Stats } from '@/types';
import CountUp from '@/components/ui/count-up';
import { Separator } from '@/components/ui/separator';

export default function Profile() {
	const [userData, setUserData] = useState<string | undefined>(localStorage.getItem('username') ?? undefined);
	const [stats, setStats] = useState<Stats | undefined>();
	console.log(stats);

	useEffect(() => {
		async function updateStats() {
			const response = await fetch(`http://localhost:8080/get-stats`, {
				headers: { 'Content-Type': 'application/json' },
				method: 'POST',
				body: JSON.stringify({ username: localStorage.getItem('username') }),
			});
			const data = await response.json();
			setStats(data);
			console.log(data);
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
									<h2 className="text-center text-lg">Total Number of Analyses</h2>
								</CardHeader>
								<CardContent>
									<CountUp
										from={0}
										to={stats?.scansNum ?? 0}
										separator=","
										direction="up"
										duration={1}
										className="count-up-text text-5xl flex justify-center"
									/>
								</CardContent>
							</Card>
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
									<div className="text-sm ml-auto mr-auto text-gray-500 w-full flex justify-center">
										<div className="flex justify-between gap-2 flex-col w-full text-base">
											<div className="flex justify-between w-full">
												<p>Left</p>
												<Separator className="flex-[0.85] mt-3" />
												<p>{Math.round(((stats?.leftBiasNum ?? 0) / Math.max(getTotalBias(), 1)) * 100)}%</p>
											</div>
											<div className="flex justify-between w-full">
												<p>Center</p>
												<Separator className="flex-[0.85] mt-3" />
												<p>
													{Math.round(((stats?.centerBiasNum ?? 0) / Math.max(getTotalBias(), 1)) * 100)}%
												</p>
											</div>
											<div className="flex justify-between w-full">
												<p>Right</p>
												<Separator className="flex-[0.85] mt-3" />
												<p>{Math.round(((stats?.rightBiasNum ?? 0) / Math.max(getTotalBias(), 1)) * 100)}%</p>
											</div>
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
									<div className="text-sm ml-auto mr-auto text-gray-500 w-full flex justify-center">
										<div className="flex justify-between gap-2 flex-col w-full text-base">
											<div className="flex justify-between w-full">
												<p>Supported</p>
												<Separator className="flex-[0.85] mt-3" />
												<p>
													{Math.round(
														((stats?.supportedClaimNum ?? 0) / Math.max(getTotalClaims(), 1)) * 100,
													)}
													%
												</p>
											</div>
											<div className="flex justify-between w-full">
												<p>Refuted</p>
												<Separator className="flex-[0.85] mt-3" />
												<p>
													{Math.round(((stats?.refutedClaimNum ?? 0) / Math.max(getTotalClaims(), 1)) * 100)}%
												</p>
											</div>
											<div className="flex justify-between w-full">
												<p>Not Enough Info</p>
												<Separator className="flex-[0.85] mt-3" />
												<p>
													{Math.round(((stats?.noInfoClaimNum ?? 0) / Math.max(getTotalClaims(), 1)) * 100)}%
												</p>
											</div>
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
