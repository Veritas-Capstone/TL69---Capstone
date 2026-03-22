import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Stats } from '@/types';

export default function UserAuth({
	setUserData,
	setStats,
}: {
	setUserData: React.Dispatch<React.SetStateAction<string | undefined>>;
	setStats: React.Dispatch<Stats | undefined>;
}) {
	const [error, setError] = useState<string | undefined>();
	const [username, setUserName] = useState<string>();
	const [password, setPassword] = useState<string>();
	const [view, setView] = useState('login');

	useEffect(() => {
		setUserName('');
		setPassword('');
		setError(undefined);
	}, [view]);

	async function login() {
		setError(undefined);
		const response = await fetch(`http://localhost:8080/user`, {
			headers: { 'Content-Type': 'application/json' },
			method: 'POST',
			body: JSON.stringify({ username: username, password: password }),
		});

		const data = await response.json();
		if (!data) {
			setError('Incorrect Password');
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
		setError(undefined);
		const response = await fetch(`http://localhost:8080/create-user`, {
			headers: { 'Content-Type': 'application/json' },
			method: 'POST',
			body: JSON.stringify({ username: username, password: password }),
		});

		const data = await response.json();
		if (!data.username) {
			setError('User already exists');
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

	if (view === 'login') {
		return (
			<Card>
				<CardHeader>
					<h1 className="text-xl">Login</h1>
				</CardHeader>
				<CardContent className="flex flex-col gap-4">
					<Input onChange={(e) => setUserName(e.target.value)} placeholder="Username" value={username} />
					<Input
						onChange={(e) => setPassword(e.target.value)}
						placeholder="Password"
						value={password}
						type="password"
					/>
					<Button onClick={login}>Login</Button>
					{error && <p className="text-red-500">{error}</p>}
					<Button onClick={() => setView('register')} variant={'link'}>
						Register
					</Button>
				</CardContent>
			</Card>
		);
	} else {
		return (
			<Card>
				<CardHeader>
					<h1 className="text-xl">Register</h1>
				</CardHeader>
				<CardContent className="flex flex-col gap-4">
					<Input onChange={(e) => setUserName(e.target.value)} placeholder="Username" value={username} />
					<Input
						onChange={(e) => setPassword(e.target.value)}
						placeholder="Password"
						value={password}
						type="password"
					/>
					<Button onClick={register}>Register</Button>
					{error && <p className="text-red-500">{error}</p>}
					<Button onClick={() => setView('login')} variant={'link'}>
						Login
					</Button>
				</CardContent>
			</Card>
		);
	}
}
