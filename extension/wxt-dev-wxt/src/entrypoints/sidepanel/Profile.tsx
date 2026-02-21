import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { PieChart, Pie, Cell, Label } from 'recharts';

export default function Profile() {
	const [userData, setUserData] = useState<string | undefined>(localStorage.getItem('username') ?? undefined);
	const [error, setError] = useState(false);
	const [username, setUserName] = useState<string>();
	const [password, setPassword] = useState<string>();
	const [stats, setStats] = useState<
		{ leftBiasNum: number; rightBiasNum: number; centerBiasNum: number } | undefined
	>();
	const [view, setView] = useState('login');

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

	useEffect(() => {
		setUserName('');
		setPassword('');
	}, [view]);

	async function login() {
		setError(false);
		const response = await fetch(`http://localhost:8080/user`, {
			headers: { 'Content-Type': 'application/json' },
			method: 'POST',
			body: JSON.stringify({ username: username, password: password }),
		});

		const data = await response.json();
		if (!data) {
			setError(true);
		} else {
			localStorage.setItem('username', data.username);
			setUserData(data.username);

			const response = await fetch(`http://localhost:8080/get-stats`, {
				headers: { 'Content-Type': 'application/json' },
				method: 'POST',
				body: JSON.stringify({ username: username }),
			});
			const data2 = await response.json();
			setStats(data2);
		}
	}

	async function register() {
		setError(false);
		const response = await fetch(`http://localhost:8080/create-user`, {
			headers: { 'Content-Type': 'application/json' },
			method: 'POST',
			body: JSON.stringify({ username: username, password: password }),
		});

		const data = await response.json();
		if (!data) {
			setError(true);
		} else {
			console.log(data);
			localStorage.setItem('username', data.username);
			setUserData(data.username);

			const response = await fetch(`http://localhost:8080/get-stats`, {
				headers: { 'Content-Type': 'application/json' },
				method: 'POST',
				body: JSON.stringify({ username: username }),
			});
			const data2 = await response.json();
			setStats(data2);
		}
	}

	function logout() {
		localStorage.clear();
		setUserData(undefined);
	}

	return (
		<>
			{!userData ? (
				view === 'login' ? (
					<Card>
						<CardHeader>
							<h1 className="text-xl">Login</h1>
						</CardHeader>
						<CardContent className="flex flex-col gap-4">
							<Input onChange={(e) => setUserName(e.target.value)} placeholder="Username" value={username} />
							<Input onChange={(e) => setPassword(e.target.value)} placeholder="Password" value={password} />
							<Button onClick={login}>Login</Button>
							{error && <p className="text-red-500">Incorrect Password</p>}
							<Button onClick={() => setView('register')} variant={'link'}>
								Register
							</Button>
						</CardContent>
					</Card>
				) : (
					<Card>
						<CardHeader>
							<h1 className="text-xl">Register</h1>
						</CardHeader>
						<CardContent className="flex flex-col gap-4">
							<Input onChange={(e) => setUserName(e.target.value)} placeholder="Username" value={username} />
							<Input onChange={(e) => setPassword(e.target.value)} placeholder="Password" value={password} />
							<Button onClick={register}>Register</Button>
							<Button onClick={() => setView('login')} variant={'link'}>
								Login
							</Button>
						</CardContent>
					</Card>
				)
			) : (
				<Card className="h-full">
					<CardHeader>
						<h1 className="text-xl">Welcome {userData}</h1>
					</CardHeader>
					<CardContent className="flex flex-col h-full">
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
								<Label
									value={`${
										(stats?.leftBiasNum ?? 0) + (stats?.rightBiasNum ?? 0) + (stats?.centerBiasNum ?? 0)
									} Checks Total`}
									position={'center'}
								/>
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
										fill={entry.name === 'Left' ? '#3b82f6' : entry.name === 'Right' ? '#ef4444' : '#8b5cf6'}
									/>
								))}
							</Pie>
						</PieChart>
						<div className="text-sm ml-auto mr-auto text-gray-500">
							<div className="flex justify-between gap-4">
								<span>
									Left:{' '}
									{Math.round(
										((stats?.leftBiasNum ?? 0) /
											((stats?.leftBiasNum ?? 0) +
												(stats?.rightBiasNum ?? 0) +
												(stats?.centerBiasNum ?? 0))) *
											100,
									)}
									%
								</span>
								<span>
									Center:{' '}
									{Math.round(
										((stats?.rightBiasNum ?? 0) /
											((stats?.leftBiasNum ?? 0) +
												(stats?.rightBiasNum ?? 0) +
												(stats?.centerBiasNum ?? 0))) *
											100,
									)}
									%
								</span>
								<span>
									Right:{' '}
									{Math.round(
										((stats?.centerBiasNum ?? 0) /
											((stats?.leftBiasNum ?? 0) +
												(stats?.rightBiasNum ?? 0) +
												(stats?.centerBiasNum ?? 0))) *
											100,
									)}
									%
								</span>
							</div>
						</div>

						<Button onClick={logout} className="mt-auto ml-auto">
							Logout
						</Button>
					</CardContent>
				</Card>
			)}
		</>
	);
}
