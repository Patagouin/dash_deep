
class Wallet:
	def __init__(self, initialAmount, orders=None):
		self.initialAmount = initialAmount
		self.orders = orders
		UEFeesDegiro = {feesFixed:0, feesPercent:0, feesPerShare:0}
		USFeesDegiro = {feesFixed:0, feesPercent:0, feesPerShare:0}
		bkr = Broker("DEGIRO", marketFees={"EU":UEFeesDegiro,"US":USFeesDegiro})

	def addOrder(order):
		self.orders += order
		# Request to broker
		

	
	def getIncome(orders):
		totalProfit=0
		totalFees=0
		for order in orders:
			order
			totalFees += (amount * fees) + (((order[3]-order[1])/order[1]) * amount) * fees
			totalProfit += ((order[3]-order[1])/order[1]) * amount
		
		return totalProfit-totalFees,totalProfit,totalFees
