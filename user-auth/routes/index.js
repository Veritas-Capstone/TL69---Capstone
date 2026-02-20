var express = require('express');
var router = express.Router();
var userSchema = require('../schemas/users');
var statsSchema = require('../schemas/stats');

router.get('/', function (req, res, next) {
	res.render('index', { title: 'Veritas DB' });
});

router.post('/create-user', function (req, res, next) {
	async function getUser() {
		return await userSchema.find({ username: req.body.username });
	}

	getUser().then((user) => {
		if (user.length === 0) {
			const newUser = new userSchema({
				username: req.body.username,
				password: req.body.password,
			});
			const newStats = new statsSchema({
				username: req.body.username,
				leftBiasNum: 0,
				rightBiasNum: 0,
				centerBiasNum: 0,
			});

			newUser
				.save()
				.then((savedUser) => {
					return newStats.save().then(() => savedUser);
				})
				.then((savedUser) => {
					res.json(savedUser);
				})
				.catch((err) => {
					console.error(err);
					res.json({ success: false, error: err.message });
				});
		} else {
			res.json({ success: false });
		}
	});
});

router.post('/user', async function (req, res, next) {
	async function getUser() {
		return await userSchema.findOne({ username: req.body.username, password: req.body.password });
	}

	getUser().then((user) => {
		res.json(user);
	});
});

router.post('/get-stats', async function (req, res, next) {
	async function getStats() {
		return await statsSchema.findOne({ username: req.body.username });
	}

	getStats().then((stats) => {
		res.json(stats);
	});
});

router.post('/stats', async function (req, res, next) {
	try {
		const { username, leftBias, rightBias, centerBias } = req.body;

		const updatedStats = await statsSchema.findOneAndUpdate(
			{ username },
			{
				$inc: {
					leftBiasNum: leftBias,
					rightBiasNum: rightBias,
					centerBiasNum: centerBias,
				},
			},
			{ new: true, upsert: true },
		);

		res.json({ success: true, stats: updatedStats });
	} catch (err) {
		console.error(err);
		res.json({ success: false, error: err.message });
	}
});

module.exports = router;
