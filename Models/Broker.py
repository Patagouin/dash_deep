class Broker:
	#marketFees : dict of dict {<marketName>:[feesFixed, feesPercent, feesPerShare],...}
	def __init__(self, name, marketFees):
		self.tickerName = name
		self.marketFees = marketFees

	@abstractmethod	
	def launchRequest(self):
		pass

class Broker_DEGIRO(Broker):
	
	def __init__(self, name, marketFees):
		__super__.init(name, marketFees)
	
	def launchRequest(order):
		pass
