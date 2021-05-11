import fundamental_indicators_provider
config = {}
company = Company('ROKU')
# Note:
#You might want to create an event loop and run within the loop:
loop = asyncio.get_event_loop()
loop.run_until_complete(fundamental_indicators_provider.get_fundamental_indicators_for_company(config, company))
print(company.fundmantal_indicators)