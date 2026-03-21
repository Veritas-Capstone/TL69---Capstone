var mongoose = require('mongoose');
var Schema = mongoose.Schema;

var Stats = new Schema({
	username: { type: String, required: true },
	leftBiasNum: { type: Number, required: true },
	rightBiasNum: { type: Number, required: true },
	centerBiasNum: { type: Number, required: true },
	supportedClaimNum: { type: Number, required: true },
	refutedClaimNum: { type: Number, required: true },
	noInfoClaimNum: { type: Number, required: true },
});

module.exports = mongoose.model('Stats', Stats);
