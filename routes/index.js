var express = require('express');
var router = express.Router();
var honeypot = require('honeypot');
//get access data
var pcap = require('pcap');
 pcap_session = pcap.createSession(device_name, options);
var pot = new honeypot(/*Add honeypot key*/);



/* GET home page. */
router.get('/', function(req, res, next) {
	var prediction;
	var packet = pcap.decode.packet(raw_packet);
	//call ml prediction algorithm
	const { spawn } = require('child_process');
    const pyProg = spawn('python', ['./prediction.py']);
    pyProg.stdout.on('data', function(data) {

       	prediction = data.toString()
        pot.query('req.ip', function(err, response){
    if (!response && prediction == "Normal") {
        res.render('index',{title:"Webpage"})
    } else {
        res.render('accden')
    }
});
    });
	
	
});

module.exports = router;
