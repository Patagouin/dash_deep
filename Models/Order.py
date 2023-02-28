		
	
class Order:
		
	def __init__(time,valueShare,type,nbShares,symbol,market,broker):
		self.time = time
		self.valueShare = valueShare
		self.type = type # 0:Buy 1:Sell
		self.nbShares = nbShares
		self.symbol = symbol
		self.broker = broker
		self.market = market
		
	def getAmount():
		return self.nbShares * self.valueShare
		
	def getFees():
		fees = self.broker.marketFees[self.market].feesFixed
		fees += self.broker.marketFees[self.market].feesPercent * self.getAmount()
		fees += self.broker.marketFees[self.market].feesPerShare * self.nbShares
		return fees



