/*	multikelly.js: Calculates precise simultaneous event Kelly stakes
	assming with no correlation and no boundary constraints

	Copyright © 2007 by Scott Eisenberg <ganchrow@sbrforum.com>. All rights reserved.
*/
	

var MaxSimult = 15;
var g_arrLineRow = new Array(MaxSimult - 1);
var g_arrResTabs = new Array(MaxSimult - 1);
var g_isDecimal = new Array(MaxSimult - 1);
var g_isEdge = new Array(MaxSimult - 1);
var g_lastSelectedTabId = 0;
var g_arrTabContents = new Array(MaxSimult - 1);
var g_arrStakePerParlaySize = new Array(MaxSimult - 1);
var g_isIE = (navigator.appName.indexOf("Microsoft") !=-1 ? 1 : 0);
var g_TextAreaDirty = false;
var g_arrStakes = new Array();
var g_fnFmt = function(x) { return x; };

function FillNumBets() {
	var mySelNumBets = $('selNumBets');
	for (var i=0; i <= MaxSimult - 1; i++) {
		mySelNumBets.options[i] = new Option(i+1,i);
	}
	mySelNumBets.options[0].selected = true;
}

function US2dec(myUS) {
	var myDec;
	myUS = parseFloat(myUS);
	if (Math.abs(myUS) < 100 || isNaN(myUS)) {
		myDec = NaN;
	} else if (myUS>0) {
		myDec = 1+myUS/100;
	} else {
		myDec = 1-100/myUS;
	}
	return myDec;
}

function dec2US(myDec) {
	var myUS;
	myDec = parseFloat(myDec);
	if (myDec <= 1 || isNaN(myDec)) {
		myUS = NaN;
	} else if (myDec < 2) {
		myUS = -100 / (myDec - 1);
	} else {
		myUS = (myDec - 1)  * 100;
	}
	return ( myUS > 0 ? "+" : "" ) + myUS.toFixed(1);
}

function prob2edge(myProb, myDec) {
	var myEdge;
	myProb = parseFloat(myProb);
	myDec = parseFloat(myDec);
	if (myProb < 0 || myProb > 1 || isNaN(myProb) || myDec < 1 || isNaN(myDec)) {
		myEdge = NaN;
	} else {
		myEdge = myDec * myProb - 1;
	}
	//return ( (100*myEdge).toPrecision(5) );
	return myEdge;
}

function edge2prob(dEdge, dOdds) {
	var dProb;
	dEdge = parseFloat(dEdge);
	dOdds = parseFloat(dOdds);
	if (dOdds < 1 || isNaN(dOdds) || isNaN(dEdge)  || dEdge+1 > dOdds || dEdge < -1) {
		myProb = NaN;
	} else {
		myProb = (dEdge + 1) / dOdds;
	}
	return myProb;
}
function GrowthVal(myGrowth,myNumBets) {
	myGrowth = parseFloat(myGrowth)/100;
	myNumBets = parseFloat(myNumBets);
	return (100 * (Math.pow(1 + myGrowth,myNumBets))).toPrecision(5) ;
}

function MakeDec(item) {
	var index = GetIdIndex(item.id);
	var myOdds = $('txtOdd_' + index).value;
	var isDecimal = $('drpOdd_' + index).value;

	if (isDecimal == "1") {
		return myOdds;
	} else {
		return US2dec(myOdds);
	}
}
function MakeEdge(item) {
	var index = GetIdIndex(item.id);
	var myDec = $('txtOddHidden_' + index).value;
	var isEdge = $('drpEdge_' + index).value;
	var myEdgeOrProb = $('txtProb_' + index).value;
	if (isEdge == "1") {
		return myEdgeOrProb;
	} else {
		return 100*prob2edge(myEdgeOrProb/100,myDec);
	}
}

function ChangeOddsFormat(item) {
	var mySelOddsType = item;
	var myTxtOdds = $('txtOdd_' + GetIdIndex(item.id));
	var isDecimal = mySelOddsType.value;
	
	if (isDecimal == "1") {	// decimal odds
		myTxtOdds.value = US2dec(myTxtOdds.value).toPrecision(5);
	} else {		// US Odds
		myTxtOdds.value = dec2US(myTxtOdds.value);
	}
}

function ChangeEdgeOrProb(item) {
	var mySelEdgeOrProb = item;
	var myTxtEdgeOrProb = $('txtProb_' + GetIdIndex(item.id));
	var myDecOdds = $('txtOddHidden_' + GetIdIndex(item.id)).value;
	
	var isEdge = mySelEdgeOrProb.value;
	if (isEdge == "1") {	// edge displayed
		myTxtEdgeOrProb.value = (100*prob2edge(myTxtEdgeOrProb.value/100,myDecOdds)).toPrecision(5);
	} else {		// prob displayed
		myTxtEdgeOrProb.value = (100*edge2prob(parseFloat(myTxtEdgeOrProb.value)/100,myDecOdds)).toPrecision(5);
	}
}

function adjOddAndEdgeLineVisibility() {
	var NumBets = parseInt($('selNumBets').value) + 1;
	var bEventsIndep = !parseInt($('selEventType').value);
	var NumParlays;
	g_arrLineRow[0].style.display = "";
	g_arrResTabs[0].style.display = "";
	for (var i = 0; i <= MaxSimult-1; i++) {
		if(i >= NumBets) {
			g_arrLineRow[i].style.display = "none";
			g_arrResTabs[i].style.display = "none";
		} else {
			g_arrLineRow[i].style.display = "";
			g_arrResTabs[i].style.display = bEventsIndep ? "" : "none";
		}
	}
	if (bEventsIndep) {
		g_arrResTabs[MaxSimult].style.display = "";
		$('txtKellyMult').disabled = false;
		$('tdStakeDescrip').innerHTML = "Stakes for parlays of size:";
	} else {
		g_arrResTabs[MaxSimult].style.display = "none";
		$('txtKellyMult').disabled = true;
		$('txtKellyMult').value = 1;
		$('tdStakeDescrip').innerHTML = "Single bet stakes:";
	}
//	autofitIframe("ToolIFrame");
	try { parent.AutoFitMe() } catch(err) {}
}

function GetInputs(arrEdges, arrOdds) {
	var NumBets = parseFloat($('selNumBets').value) + 1;

	var arrEdges = new Array(NumBets);
	var arrOdds = new Array(NumBets);

	for(var i = 0; i <= NumBets-1; i++) {
		var textLineOddHiddenId = "txtOddHidden_" + i;
		var textLineEdgeHiddenId = "txtEdgeHidden_" + i;
		arrOdds[i] = $(textLineOddHiddenId).value;
		arrEdges[i] = parseFloat($(textLineEdgeHiddenId).value)/100;
	}
	return {arrEdges: arrEdges, arrOdds: arrOdds};
}

function CreateOddAndEdgeRows() {
	var onChangeEvent = '';
	
	for (var i = 0; i<= MaxSimult-1; i++) {
		
		var textLineOddDropId = "drpOdd_" + i;
		var textLineOddBoxId = "txtOdd_" + i;
		var textLineOddHiddenId = "txtOddHidden_" + i;		
		var textLineEdgeDropId = "drpEdge_" + i;
		var textLineProbBoxId = "txtProb_" + i;
		var textLineEdgeHiddenId = "txtEdgeHidden_" + i;
		
		var lblOddsType = "lblOddsType_" + i;
		var lblEdgeOrProb = "lblEdgeOrProb_" + i;
		
		var rowId = "rowLine_" + i;
		
		document.write("<tr id='" + rowId + "'>");
		document.write("<td id='" + lblOddsType + "'><select class='lselect' name='" + textLineOddDropId + "' id='" + textLineOddDropId + "' onChange='ChangeOddsFormat(this);'><option value='1'>Decimal Odds:</option><option value='0' selected>US Odds:</option></select></td>");		
		document.write("<td align='right'><input class='rtext' type='text' value='-110.0' name='" + textLineOddBoxId + "' id='" + textLineOddBoxId + "' onChange='$(" + textLineOddHiddenId + ").value=MakeDec(this);$(" + textLineEdgeHiddenId + ").value=MakeEdge(this);' onBlur='$(" + textLineOddHiddenId + ").value=MakeDec(this);$(" + textLineEdgeHiddenId + ").value=MakeEdge(this);'><input type='hidden' value='" + (1 + 100/110) + "' name='" + textLineOddHiddenId + "' id='" + textLineOddHiddenId + "'></td>");
		document.write("<td>&nbsp;</td>");
		document.write("<td>&nbsp;</td>");
		document.write("<td id='" + lblEdgeOrProb + "'><select class='lselect' name='" + textLineEdgeDropId + "' id='" + textLineEdgeDropId + "' onChange='ChangeEdgeOrProb(this);'><option value='1'>Edge:</option><option value='0' selected>Win Prob:</option></select></td>");
		document.write("<td align='right'><input class='rtext' type='text' name='" + textLineProbBoxId + "' id='" + textLineProbBoxId + "' value='55.0000' onChange='$(" + textLineEdgeHiddenId + ").value=MakeEdge(this);' onBlur='$(" + textLineEdgeHiddenId + ").value=MakeEdge(this);'><input type='hidden' value='5' name='" + textLineEdgeHiddenId + "' id='" + textLineEdgeHiddenId + "'></td>");
		document.write("<td class='ltext'>%</td>");
		document.write("</tr>");
		
		g_arrLineRow[i] = $(rowId);
	}
}

function CreateKellyStakeDivs() {
	document.write("<tr><td colspan=7 align=center><table cellspacing=0 cellpadding=0>");
	document.write("<tr><td align=left class=formtext id=\"tdStakeDescrip\">Stakes for parlays of size:</td></tr>");		
	document.write("<tr id=tabs><td align=left>");
	document.write("<table cellpadding=4 cellspacing=0><tr>");
	for (var i = 0; i<= MaxSimult; i++) {
		var divTabId = "tabRes_" + i;
		var divResId = "divRes_" + i;
		var rowResId = "rowRes_" + i;
		document.write("<td class='tabNotSelected' id='" + divTabId + "' onClick='SelectTab(" + i + ", true)'>" + (i!=MaxSimult ? i+1 : 'All' ) + "</td>");
		g_arrResTabs[i] = $(divTabId);
	}
	document.write("</tr>");
	document.write("</table></td></tr>");
	document.write("<tr><td colspan='7' align='center'><textarea class='restarea' id='taResults' name='taResults' cols=48 rows=6 onChange='if (this.value.search(/Daisy/i) > -1) { this.value = this.value.replace(/Daisy/i,\"\"); var w = window.open(\"http://bettingtools.sbrforum.com/etc/daisy.mp3\", \"IBM 704 Daisy\", \"width=115,height=105,resizable=no,status=no\"); return false;} g_TextAreaDirty=true; if (this.value == \"Daisy\") window.location=\"http://bettingtools.sbrforum.com/etc/daisy.mp3\"; SelectTab(g_lastSelectedTabId,true);' onBlur=''  readonly></textarea></tr></td>");
	document.write("<tr><td colspan='7' id=tdParlaySizeDescrip class='formtext'></td></tr></table>");
	SelectTab(0,false);
	DisplayStakes(new Array(), function() {} );
}

function SelectTab(selectTabId, SaveCurrentTab) {
	var thisSizeName;
	$('taResults').readOnly = false;
	if (selectTabId == MaxSimult) {
		thisSizeName = 'All bets';
		$('taResults').readOnly = true;
	} else if (selectTabId  > 0) {
		thisSizeName = (selectTabId + 1) + '-team parlays';
	} else {
		thisSizeName = 'Single bets';
	}
	if (SaveCurrentTab == true) {
		var Bankroll = UnFmtNumber($('txtBankroll').value);

		if (g_lastSelectedTabId != MaxSimult) {
			g_arrTabContents[g_lastSelectedTabId] = $('taResults').value;
		}

		if (selectTabId == MaxSimult){
			g_arrTabContents[MaxSimult] = '';
			g_arrTabContents[MaxSimult] = g_arrTabContents.join('\n').replace(/\n+/gm,'\n');;
			g_arrTabContents[MaxSimult] = g_arrTabContents[MaxSimult].replace(/\n$/m,'');;
		}
		if (g_TextAreaDirty == true) {
			var arrNamesStakes = g_arrTabContents[g_lastSelectedTabId].split('\n');
			g_arrStakePerParlaySize[MaxSimult] -= g_arrStakePerParlaySize[g_lastSelectedTabId];
			g_arrStakePerParlaySize[g_lastSelectedTabId] = 0;
			for (var i=0; i<=arrNamesStakes.length-1; i++) {
				var thisNameStake = arrNamesStakes[i].split('\t');
				var ParlayNumber = 0;
				var arrParlayName = thisNameStake[0].split(/\+/);
				for(n in arrParlayName) {
					ParlayNumber += Math.pow(2,arrParlayName[n]-1);
				}

				thisNameStake[1] = UnFmtNumber(thisNameStake[1])/Bankroll;
				g_arrStakePerParlaySize[g_lastSelectedTabId] += thisNameStake[1];
				if(Math.abs(g_arrStakes[ParlayNumber] - thisNameStake[1]) > 1e-6 ) {
					g_arrStakes[ParlayNumber] = thisNameStake[1];
				}
			}
			g_arrStakePerParlaySize[MaxSimult] += g_arrStakePerParlaySize[g_lastSelectedTabId];
		}
	}

	g_lastSelectedTabId = selectTabId;
	for (var i = 0; i<= MaxSimult; i++) {
		var divTabId = "tabRes_" + i;
		$(divTabId).className = selectTabId == i ? 'tabSelected' : 'tabNotSelected';
	}
	$('taResults').value = g_arrTabContents[selectTabId] ? g_arrTabContents[selectTabId] : '';
	if (g_arrStakePerParlaySize[selectTabId] == undefined) {
		$('tdParlaySizeDescrip').innerHTML = thisSizeName;
		$('taResults').rows = 1;
	} else {
		if (g_arrStakePerParlaySize[selectTabId] > 0) {
			$('tdParlaySizeDescrip').innerHTML = 'Total stake for all ' + thisSizeName.toLowerCase() + ': ' + g_fnFmt(UnFmtNumber(g_arrStakePerParlaySize[selectTabId]));
		} else {
			$('tdParlaySizeDescrip').innerHTML = '';
		}
		$('taResults').rows = Math.max(g_arrTabContents[selectTabId].split('\n',7).length-1+g_isIE,1);
	}
	g_TextAreaDirty  = false;
}

function GetIdIndex(name) {
	var id = name.split("_");
	return id[1];
}

function bin2dec(b) {
	return (parseInt(b,2));
}

function dec2bin(d,sig) {
	var b = d.toString(2);
	while (b.length < sig ) {
		b = '0' + b;
	}
	return b;
}

function isArray(obj){
	return typeof(obj) != "object" ? false : ( typeof(obj.length)=="undefined" ? false : true);
}
function StrRev(s) {
	if (!s) return '';
	var sRevStr='';
	for (var i = s.length-1; i>=0; i--) {
		sRevStr+= s.charAt(i);
	}
	return sRevStr;
}

function bitIMP(a,b) {
	// bitwise impliction
	return (~a | b);
}

function UnFmtNumber(num) {
	num = '' + num;
	num = num.replace(/[,\$\s]/g, '');
	if(num.indexOf('%') > -1) {
		num = num.replace(/%/g, '');
		num = parseFloat(0+num/100);
	} else {
		if (num == undefined || isNaN(num) || num=='') num = "0";
		num = parseFloat(num);
	}
	return num;
}

function DisplayStakes(objStakes,fnFmt) {
	var arrNames = objStakes.arrNames;
	var arrStakes = objStakes.arrStakes;

	for (var i=0; i <= MaxSimult; i++) {
		g_arrTabContents[i] = '';
		g_arrStakePerParlaySize[i] = 0;
	}

	if (isArray(arrStakes)) {
		for (var i=1; i <= arrStakes.length-1; i++) {
			if (arrNames[i] != undefined) {
				var ParlaySize = arrNames[i].split('+').length - 1;
				var StakeSize = arrStakes[i];
				g_arrStakePerParlaySize[ParlaySize] += StakeSize;
				g_arrTabContents[ParlaySize] += "\n" + arrNames[i] + "\t" +fnFmt(StakeSize);
				g_arrStakePerParlaySize[MaxSimult] += StakeSize;
			}
		}
	}

	for (var i=0; i <= MaxSimult - 1; i++) {
//		g_arrStakePerParlaySize[i] = fnFmt(parseFloat(g_arrStakePerParlaySize[i]));
		g_arrTabContents[i] = g_arrTabContents[i].replace(/^\n/m,'');
		if (g_arrTabContents[i].length > 0) {
			g_arrTabContents[MaxSimult] += '\n' + g_arrTabContents[i];
		}
	}
	g_arrTabContents[MaxSimult] = g_arrTabContents[MaxSimult].replace(/^\n/m,'');
//	g_arrStakePerParlaySize[MaxSimult] = fnFmt(parseFloat(g_arrStakePerParlaySize[MaxSimult]));
	g_arrTabContents[i] = g_arrTabContents[MaxSimult].replace(/^\n/m,'');
	g_fnFmt = fnFmt;
	SelectTab(0, false);
}

function ParlaySize(lParlay,lBets) {
	var sParlay = dec2bin(lParlay,0);
	var lSize = 0;
	for (var i = 0; i < sParlay.length; i++) {
		lSize += parseInt(sParlay.charAt(i));
	}
	return lSize;
}

function calcMutExKelly(a_dEdge, a_dOdds, dKellyMult)  {
	var lSingles;
	var a_sParlayNames;
	var a_dRealKellyStakes;

	if (dKellyMult == undefined || dKellyMult <= 0 || isNaN(dKellyMult)) dKellyMult = 1;
	dKellyMult = parseFloat(dKellyMult);
	
	if (!isArray(a_dEdge) || !isArray(a_dOdds) ) {
		var err = "calcMutExKelly: Odds and Edge arguments must both be arrays";
		alert(err);
		return err;
	}
	lSingles = a_dOdds.length;
	if(lSingles != a_dEdge.length) {
		var err = "calcMutExKelly: Edge (size=" + a_dEdge.length + ") and odds (size=" + lSingles + ") arrays are different sizes";
		alert(err);
		return err;
	}
	a_dRealKellyStakes = new Array(Math.pow(2,lSingles)-1);
	a_sParlayNames = new Array(Math.pow(2,lSingles)-1);
	var oSortedByEdge = new Array(lSingles-1);
	var dTotProb = 0;
	for (var i=0; i<=lSingles-1;i++) {
		var mydProb = edge2prob(a_dEdge[i],  a_dOdds[i]);
		dTotProb += mydProb ;
		oSortedByEdge[i] = { n: i, dEdge: a_dEdge[i], dOdds: a_dOdds[i], dProb: mydProb};
	}
	if(dTotProb > 1 + 1e-6) {
		var err = "calcMutExKelly: Sum of probabilities of mutually exclusive outcomes (" + dTotProb + ") may not be > 1";
		alert(err);
		return err;
	}

	var fnSortByEdge = function(a,b) { return(b.dEdge - a.dEdge); } ;
	oSortedByEdge.sort(fnSortByEdge);
	var dMinResult = 1, dOverround = 0, dSumProb = 0, dSumOddsRecip = 0;

	for (var i = 0; i<=lSingles-1; i++) {
		dSumProb += oSortedByEdge[i].dProb;
                if ( dSumProb > 1 ) dSumProb = 1; // due to rounding error probability may erroneously be slightly > 1
		dOverround += 1 / oSortedByEdge[i].dOdds;
		var dProposedMinResult = (1-dSumProb) / (1-dOverround );
		if (dProposedMinResult > 0 && dProposedMinResult < dMinResult) {
			dMinResult = dProposedMinResult ;
		}
	}
	for (var i = 0; i<=lSingles-1; i++) {
		if (dOverround < 1 && dSumProb >= 1 - 1e-7 ) {
			a_dRealKellyStakes[Math.pow(2,oSortedByEdge[i].n)] = oSortedByEdge[i].dProb;
		} else {
			a_dRealKellyStakes[Math.pow(2,oSortedByEdge[i].n)] = Math.max(0, oSortedByEdge[i].dProb - dMinResult / oSortedByEdge[i].dOdds);
		}
		a_sParlayNames[Math.pow(2,oSortedByEdge[i].n)] = ''+(1+oSortedByEdge[i].n);
	}
	g_arrStakes = a_dRealKellyStakes;
	return {arrNames: a_sParlayNames, arrStakes: a_dRealKellyStakes};
}

function calcKelly(a_dEdge, a_dOdds, dKellyMult, bMutuallyExclusive)  {
	
	
	if (bMutuallyExclusive == true) return calcMutExKelly(a_dEdge, a_dOdds, dKellyMult);

	var lBets, lSingles;
	var a_sParlayNames;
	var a_dSingKellyStakes, a_dRealKellyStakes;
	var aa_lParlayMap;

	if (dKellyMult == undefined || dKellyMult <= 0 || isNaN(dKellyMult)) dKellyMult = 1;
	dKellyMult = parseFloat(dKellyMult);
	
	if (!isArray(a_dEdge) || !isArray(a_dOdds) ) {
		var err = "calcKelly: Odds and Edge arguments must both be arrays";
		alert(err);
		return err;
	}
	lSingles = a_dOdds.length;
	if(lSingles != a_dEdge.length) {
		var err = "calcKelly: Edge (size=" + a_dEdge.length + ") and odds (size=" + lSingles + ") arrays are different sizes";
		alert(err);
		return err;
	}
	lBets = Math.pow(2,lSingles) - 1;
	a_dSingKellyStakes = new Array(lSingles);
	a_dRealKellyStakes = new Array(lBets+1);
	a_sParlayNames = new Array(lBets+1);
	aa_lParlayMap = new Array(lSingles);
	for(var i = 0; i <= lSingles-1; i++) {
		if ( a_dOdds[i] <= 1 ) {
			var err = "calcKelly: Bet # " + i + " odds (" + a_dOdds[i] + ") <= 1";
			alert(err);
			return err;
		} else if (a_dEdge[i] > a_dOdds[i] - 1 ) {
			var err = "calcKelly: Bet # " + (i+1) + " edge (" + a_dEdge[i] + ") > odds-1 (" + (a_dOdds[i] - 1) + ")";
			alert(err);
			return err;
		}
		var odds = parseFloat(a_dOdds[i]);
		var edge = parseFloat(a_dEdge[i]);
		var prob = (1 + edge) / odds;
		var win  = odds - 1;
		a_dSingKellyStakes[i] = Math.max( 0,
							( Math.pow(win*prob, dKellyMult) -
							  Math.pow(1 - prob, dKellyMult) ) /
							( Math.pow(win*prob, dKellyMult) +
							  win * Math.pow(1 - prob, dKellyMult)
							)
						);

		aa_lParlayMap[i] = new Array();
	}
	for(var i = 1; i <= lBets; i++) {
		var parlaySize = ParlaySize(i);
		aa_lParlayMap[parlaySize-1].push(i);
		//document.writeln(i + " => " + dec2bin(i,lSingles) + " => " + parlaySize + " => " + aa_lParlayMap[parlaySize-1]);
	}
	//document.writeln("");
	for(var s = lSingles; s >= 1; s--) {
		// s = parlay size (long)
		var iLimit = aa_lParlayMap[s-1].length;
		for (i = 0; i <iLimit; i++ )
		{
			var p_n = aa_lParlayMap[s-1][i];
			// p_n = parlay number (long)
			a_dRealKellyStakes[p_n] = 1;
			a_sParlayNames[p_n] = "";
			for(var k = 0; k <= lSingles-1; k++)
			{
				if (bitIMP(Math.pow(2,k),p_n) == -1 ) 
				{
					a_dRealKellyStakes[p_n] *= a_dSingKellyStakes[k];
					a_sParlayNames[p_n] += ((k+1) + "+");
				}

			}
			for(var ss = s+1; ss <= lSingles ; ss++) 
			{
			
				var ssLimit= aa_lParlayMap[ss-1].length;
				for (ii = 0; ii <ssLimit; ii++ )
				{
					var pp = aa_lParlayMap[ss-1][ii];
					if (bitIMP(p_n,pp) == -1 ) {
						a_dRealKellyStakes[p_n] -= a_dRealKellyStakes[pp];
					}
				}
			}
			a_sParlayNames[p_n] = a_sParlayNames[p_n].substring(0,a_sParlayNames[p_n].length-1);	// remove trailing underscore
			//document.writeln(p_n + "=" + dec2bin(p_n,lSingles) + " => " + (100*a_dRealKellyStakes[p_n]).toPrecision(5) + "% => " + a_sParlayNames[p_n]);
		}
	}
	//alert(a_sParlayNames.length);
	g_arrStakes = a_dRealKellyStakes;
	return {arrNames: a_sParlayNames, arrStakes: a_dRealKellyStakes};
}

function calcMutExEVG(a_dEdge, a_dOdds)  {
	var lSingles;
	var dEV = 0, dEG = 1, dTotalProb = 0, dTotalStake = 0;

	if (!isArray(a_dEdge) || !isArray(a_dOdds)) {
		var err = "calcMutExEVG: Odds, Edge, and Stakes arguments must all be arrays";
		alert(err);
		return err;
	}
	lSingles = a_dOdds.length;
	var a_dStakes = new Array(lSingles-1);
	var a_dProb = new Array(lSingles);

	if(lSingles != a_dEdge.length) {
		var err = "calcMutExEVG: Edge and Odds arrays are different sizes";
		alert(err);
		return err;
	}

	for (var i = 0; i <= lSingles-1; i++) {
		a_dProb[i] = edge2prob(a_dEdge[i],a_dOdds[i]);
		a_dStakes[i] = g_arrStakes[Math.pow(2,i)];
		dTotalStake += a_dStakes[i];
		dTotalProb += a_dProb[i];
	}

	for (var i = 0; i <= lSingles-1; i++) {
		dEV += (a_dStakes[i] * a_dOdds[i] - dTotalStake) * a_dProb[i];
		dEG *= Math.pow(1 + a_dStakes[i] * a_dOdds[i] - dTotalStake, a_dProb[i]);
	}
	dEV -= dTotalStake * (1-dTotalProb)
	if(dTotalProb < 1) dEG *= Math.pow(1-dTotalStake, 1-dTotalProb);
	dEG -= 1;
	return {dEV : dEV, dEG : dEG};
}

function calcEVG(a_dEdge, a_dOdds, bMutuallyExclusive)  {
	// calculates expected value (P&L) and expected growth
	SelectTab(g_lastSelectedTabId,true);	// save current tab
	if (bMutuallyExclusive == true) return calcMutExEVG(a_dEdge, a_dOdds);
	var a_dStakes = g_arrStakes;
	var lSingles, lBets;
	var dTotalStake = 0, dEV = 0, dEG = 1;
	var a_dProb, aa_lParlayMap, a_dOutcomeProbs, a_dParlayOdds, a_dWinAmt;

	if (!isArray(a_dEdge) || !isArray(a_dOdds) || !isArray(a_dStakes)) {
		var err = "calcEVG: Odds, Edge, and Stakes arguments must all be arrays";
		alert(err);
		return err;
	}
	lSingles = a_dOdds.length;
	if(lSingles != a_dEdge.length) {
		var err = "calcEVG: Edge and Odds arrays are different sizes";
		alert(err);
		return err;
	}
	lBets = Math.pow(2,lSingles) - 1;
	if(lBets != a_dStakes.length-1 || (a_dStakes[0] != undefined && a_dStakes[0] != 0) ) {
		var err = "calcEVG: Stakes array (size=" + a_dStakes.length + ") must contain 2^(Num Bets)  = " + (lBets+1) + " and first element (" + a_dStakes[0] + ") must be null";
		alert(err);
		return err;
	}
	a_dStakes[0] = 0;
	a_dProb = new Array(lSingles);
	aa_lParlayMap = new Array(lBets);
	a_dOutcomeProbs = new Array(lBets);
	a_dParlayOdds = new Array(lBets);
	a_dWinAmt =  new Array(lBets);
	for (var s = 0; s <= lSingles-1; s++) {
		a_dProb[s] = edge2prob(a_dEdge[s],a_dOdds[s]);
		aa_lParlayMap[s+1] = new Array();
	}
	aa_lParlayMap[0] = new Array();

	for (var p = 0; p <= lBets; p++) {
		var parlaySize = ParlaySize(p);
		aa_lParlayMap[parlaySize].push(p);
		dTotalStake += a_dStakes[p];
		var sParlay = dec2bin(p,lSingles);
		a_dOutcomeProbs[p] = 1;
		a_dParlayOdds[p] = 1;
		for (var s = 0; s <= lSingles-1; s++) {
			if(sParlay.charAt(lSingles-1-s) == "0") {
				a_dOutcomeProbs[p] *= (1-a_dProb[s]);
			} else {
				a_dOutcomeProbs[p] *= a_dProb[s];
				a_dParlayOdds[p] *= a_dOdds[s];
			}
		}
		a_dWinAmt[p] = a_dStakes[p] * a_dParlayOdds[p];
		//document.writeln(StrRev(dec2bin(p,lSingles)) + " => " + a_dOutcomeProbs[p] + " => " + a_dParlayOdds[p] + " => " + a_dWinAmt[p] );
	}
	
	for(var s = lSingles; s >= 0; s--) {
	
		var iLimit = aa_lParlayMap[s].length;
		for (i = 0; i <iLimit; i++ )
		{
			var p = aa_lParlayMap[s][i];
			var dThisRet = 0;
			dThisRet -= dTotalStake;
			dThisRet +=  a_dWinAmt[p];	// the parlay associated with current outcome looped over will be only parlay of this size to win
			for(var ss = s-1; ss >= 1; ss--) {
				var ssLimit=aa_lParlayMap[ss].length;
				for (ii = 0; ii < ssLimit; ii++ )
				{
					var pp = aa_lParlayMap[ss][ii];
					if (bitIMP(pp,p) == -1 ) {
						dThisRet +=  a_dWinAmt[pp];
					}
				}
			}
			dEV += isNaN(a_dOutcomeProbs[p] * dThisRet) ? 0 : a_dOutcomeProbs[p] * dThisRet; 
			dEG *= isNaN(Math.pow(1+dThisRet,a_dOutcomeProbs[p])) ? 1 : Math.pow(1+dThisRet,a_dOutcomeProbs[p]);
		}
	}
	dEG -= 1;
	return {dEV : dEV, dEG : dEG};
}

function calcTotalGrowth(dRate,lPeriods) {
	var dGrowth = Math.pow(1+dRate,lPeriods);
	return dGrowth;
}
/*

function $() {
	var elements = new Array();
	for (var i = 0; i < arguments.length; i++) {
		var element = arguments[i];
		if (typeof element == 'string')
			element = document.getElementById(element);
		if (arguments.length == 1)
			return element;
		elements.push(element);
	}
	return elements;
}
*/
