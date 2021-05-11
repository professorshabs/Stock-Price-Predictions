import fundamental_indicators_provider as fip
config = {}
company = fip.Company('ROKU')
# Note:
#You might want to create an event loop and run within the loop:
loop = fip.asyncio.get_event_loop()
loop.run_until_complete(fip.get_fundamental_indicators_for_company(config, company))
print(company.fundmantal_indicators)