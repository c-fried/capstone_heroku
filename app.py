import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np
import pandas as pd
import json
import os
import pickle
import joblib
import sklearn

import plotly.graph_objects as go
import plotly.express as px
from plotly.figure_factory import create_distplot

# from tensorflow.keras.models import load_model
from baseball_support import (
	Simulator, PlayerFinder, DataStorage, load_preprocessors, shuffle_lst
	)

# SETUP
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # prevent CUDA from using GPU
# MODEL_PATH = 'trained_model_1_layer_no_weights.h5'
# model = load_model(MODEL_PATH)
SGD_PATH = 'sgd.pkl'
model = joblib.load(SGD_PATH)

X_preprocessor, y_preprocessor = load_preprocessors()
MLB = pickle.load(open('./data/populated_league_sm.pkl', 'rb'))
pf = PlayerFinder()
simulator = Simulator(model, MLB, X_preprocessor, y_preprocessor, pf)

storage = DataStorage()
print('All loaded.')

# Layout variables.
# player_options = [
# 	{'label': f"{d['last']}, {d['first']}: {d['play_debut'].year}", 
# 	 'value': idx}
# 	for idx, d in pf.player_df.iterrows()
#  ]
with open(r"player_options.json", "r") as read_file:
    player_options = json.load(read_file)


def player_dropdown(_id, player_id=''):
	return dcc.Dropdown(
		id=_id,
		options=player_options, 
		placeholder='Last, First: Debut Year.',
		value=player_id
 	)


yankees98 = [
	'posaj001',
	'martt002',
	'knobc001',
	'bross001',
	'jeted001',
	'curtc001',
	'willb002',
	'oneip001',
	'strad001'
]
maddux = 'maddg002'

lineup_selector = dbc.Col([
	dbc.Row(html.H4('Batting Order'))] + [
	dbc.Row(
		dbc.Col(player_dropdown(f'hitter-{n}-input', player_id)), 
		justify='center')
	for n, player_id in zip(range(1, 10), yankees98)],
	width=5
)

pitcher_selector = dbc.Col([
	dbc.Row(html.H4('Opposing Pitcher')),
	dbc.Row(dbc.Col(player_dropdown('pitcher-input', maddux)), justify='center')
	],
	width=5
)

inning_slider = dbc.FormGroup([
	dbc.Label('Innings to Simulate'),
	dcc.RangeSlider(
		id='inning-number', 
		min=1, 
		max=9, 
		step=1, 
		value=[1, 9], 
		allowCross=False,
		marks={n:str(n) for n in range(1,10)})
])

double_header_selector = dbc.FormGroup([
	dbc.RadioItems(
		id='doubleheader',
		options=[
			{'label': 'Single Game', 'value': 0},
			{'label': 'Doubleheader: Game 1/2', 'value': 1},
			{'label': 'Doubleheader: Game 2/2', 'value': 2}
			],
		value=0
		)
])

home_away_selector = dbc.FormGroup([
	# dbc.Label('Home or Away'),
	dbc.RadioItems(
		id='inning-half',
		options=[
			{'label': 'Home', 'value': 1},
			{'label': 'Away', 'value': 0}
			],
		value=1,
		inline=True
		)
])

day_night_selector = dbc.FormGroup([
	dbc.Label('Time of Game'),
	dbc.RadioItems(
		id='day-night',
		options=[
			{'label': 'Day', 'value': 'day'},
			{'label': 'Night', 'value': 'night'}
			],
		value='night',
		inline=True
		)
])

career_stats_selector = dbc.FormGroup([
	# dbc.Label('Use Career Stats'),
	dbc.Checklist(
		id='career-stats',
		options=[{'label': 'Use Player Career Stats', 'value': True}],
		value=[True],
		switch=True
		)
])

temp_selector = dbc.FormGroup([
	dbc.Label('Temp. (°F)'),
	dbc.Input(id='temp', type='number', value=-1, min=-1, step=1)
])

wind_selector = dbc.FormGroup([
	dbc.Label('Wind Speed (mph)'),
	dbc.Input(id='wind', type='number', value=-1, min=-1, step=1)
])

field_condition_selector = dbc.FormGroup([
	dbc.Label('Field Conditions'),
	dbc.Select(options=[
		{'label': x.title(), 'value': x} 
		for x in ('unknown', 'dry', 'wet', 'damp', 'soaked')
		],
		id='conditions',
		value='unknown')
])

precipitation_selector = dbc.FormGroup([
	dbc.Label('Precipitation'),
	dbc.Select(options=[
		{'label': x.title(), 'value': x} 
		for x in ('unknown', 'none', 'rain', 'drizzle', 'showers', 'snow')
		],
		id='precip',
		value='unknown')
])

attendance_selector = dbc.FormGroup([
	dbc.Label('Attendance'),
	dbc.Input(id='attendance', type='number', value=-1, min=-1, step=1)
])

# Input layout.
INPUT_DIV = html.Div(id='input_form', children=[
	dbc.Form([
		dbc.Row([
			dbc.Col([	# left col
				dbc.Row(
					dbc.Col(	# header
						html.H2(
							'Simulation Info'
							), 
						width='auto'), 
					justify='center'
					),
				dbc.Row([	# lineup/pitcher
					lineup_selector,
					pitcher_selector
					], 
					justify='center'),
				dbc.Row(	# inning start
					dbc.Col(inning_slider, width=8),
					justify='center'
					), 
				dbc.Row([	# homeaway/dblheader/careerstats
					dbc.Col(home_away_selector, width=4),
					dbc.Col(double_header_selector, width=4),
					dbc.Col(career_stats_selector, width=4)
					],
					justify='between')]),
			dbc.Col([	# right col
				dbc.Row(	# Header
					dbc.Col(
						html.H3('Game-Time Conditions'), 
						width='auto'), 
					justify='center'),
				dbc.Row([	# daynight/temp/wind
					dbc.Col(day_night_selector, width=4),
					dbc.Col(temp_selector, width=4),
					dbc.Col(wind_selector, width=4)
					],
					justify='center'),
				dbc.Row([	# field/precip
					dbc.Col(field_condition_selector, width='auto'),
					dbc.Col(precipitation_selector, width='auto')
					],
					justify='center'), 
				dbc.Row(	# attendance
					dbc.Col(attendance_selector, width='auto'),
					justify='center'
					)])
			],
			justify='center')
		]),
	html.Hr()
	])

# Simulator div.
SIMULATE_BRANCH_DIV = html.Div(id='simulate-branch', children=[
	dbc.Row(
		dbc.Col(
			html.H2('Simulate: Single Game.'),
			width='auto'),
		justify='center'
		),
	dbc.Row(
		dbc.Col(
			dbc.Button(
				'Run Single Simulation', 
				id='submit-single', 
				color='primary'
				),
			width='auto'),
		justify='center'
		),
	html.Div(id='single-output', children=[
		dcc.Loading(
			id='loading-output-single',
			children=[dcc.Graph(id='single-output-graph')],
			type='default'),
		dcc.Markdown(id='single-output-md')
		])
])

SIMULATE_MULTI_DIV = html.Div(id='simulate-multi', children=[
	dbc.Row(
		dbc.Col(
			html.H2('Simulate: Multiple Games.'),
			width='auto'),
		justify='center'
		),
	dbc.Row(
		dbc.Col([
			dbc.FormGroup([
				dbc.Label('Number of Simulations (max: 200)'),
				dbc.Input(
					id='num-simulations', 
					type='number', value=30, min=2, max=200, step=1)
				]),
			dbc.Button(
				'Run Multi-Branch Simulation', 
				id='submit-multi', 
				color='primary'
				)],
			width='auto'),
		justify='center'
		),
	html.Div(id='multi-output', children=[
		dcc.Loading(
			id='loading-output-multi',
			children=[dcc.Graph(id='multi-output-graph')],
			type='default'),
		dcc.Markdown(id='multi-output-md')
		])
])

# Optimizer div.
OPTIMIZE_DIV = html.Div(id='optimize', children=[
	dbc.Row(
		dbc.Col([
			dbc.Row(
				html.H2('Optimize Batting Order'), 
				justify='center'),
			dbc.Row(
				html.H4('Automatically sort inputted lineup'), 
				justify='center')
			],
			width='auto'),
		justify='center'
		),
	dbc.Row(
		dbc.Col([
			dbc.FormGroup([
				dbc.Label('Simulations per Order Shuffle'),
				dbc.Input(
					id='sims-per-order',
					type='number', value=25, min=2, max=200, step=1),
				dbc.Label('Show Live Updates of Optimizer'),
				dbc.Checkbox(
					id='toggle-interval',
					checked=True
					)]),
			dbc.Button(
				'Optimize Given Lineup',
				id='submit-optimize',
				color='primary'
				),
			dcc.Interval(
				id='optimze-interval-component',
				disabled=False
				)],
			width='auto'),
		justify='center'
		),
	dbc.Row(
		dcc.Loading(
			id='loading-output-optimize',
			children=[dcc.Markdown(id='loading-optimize')],
			type='default'),
		justify='center'),
	dbc.Row([
		dbc.Col(dcc.Markdown(id='lineup-locks-output'), width=5),
		dbc.Col(dcc.Markdown(id='trying'), width=5)
		],
		justify='center')
	])

# Main container: Heading - title, subtitle, info, tabs.
HEADING_DIV = html.Div(id='heading', children=[
	dbc.Row(
		dbc.Col(
			html.H1('Major League Baseball Simulator'), 
			width='auto'), 
		justify='center'
		),
	dbc.Row(
		dbc.Col(
			html.H3('Simulating Gameplay using Neural Networks'),
			width='auto'),
		justify='center'),
	dbc.Row(
		dbc.Col(
			html.P("""
				Beginning in the mid-1800s (and still continuing strong today), 
				baseball is clearly one of the staples of American culture. With 
				player and game data going back well over 100 years, it's a ripe 
				domain to study.
				"""),
			width=8
			),
		justify='center'),
	dbc.Row(
		dbc.Col(
			html.P("""
				This app uses an SGD Logistic Regression model (built in 
				Sklearn) and "event" data that I've collected and engineered 
				from Retrosheet.org.
				My goal was to make a model that could predict the outcome of an
				"event" (at-bat) during a game. There are 10 outcomes of an 
				event, such as ['single', 'strikeout', 'home-run', 'walk', etc].
				The games are reconstructed (without baserunning considerations)
				by taking the results of each play and inputting them into the 
				game-state. Amazingly, without any knowledge of what
				"runs-scored" are, the model's simulations are very good 
				representations (as validated by season-simulations of past
				years).
				"""),
			width=8
			),
		justify='center'),
	dbc.Row(
		dbc.Col(
			html.P(""" 
				There are two aspects to factor in: player-data and game-data.
				Every player (pitcher and hitter) since 1950 has been accounted
				for, as well as their at-bat-to-at-bat stats. The Hitter vs
				Pitcher dynamic is a huge factor in how the model predicts the 
				outcomes. There is also game data (temperature / attendance) 
				which the model uses to influence its predictions. Finally, 
				there is dynamic data of what has happened just before in an
				inning. For example, if the previous 2 hitters just walked, how
				does that impact the liklihood of the next outcome?
				"""),
			width=8
			),
		justify='center'),
	html.Hr(),
	dbc.Row(
		dbc.Col(
			dcc.Markdown("""### The app has three main functions:
				\n\n- **Lineup Optimizer:**
				\n > A tool that will try to find the best batting-order from a 
				list of hitters. The best batting order will be the one that has
				the highest "expected runs scored" over each iteration through 
				the search. 
				\n > *The searching algorithm is in no way exhaustive, 
				but returns some very interesting results. The greater the
				number of **"Simulations per Order Shuffle"**, the more likely
				you are to getting consistent results.*
				\n\n- **Single Game Simulation:**
				\n > Run a detailed simulation where you can see every play in a 
				game / shortened-simulation.
				\n\n- **Multi-Game Simulation:**
				\n > Run multiple simulations where the same lineup is facing the
				same lineup. This is useful to see how the lineup is expected to 
				perform over a long stretch of time.
				"""),
			width=8
			),
		justify='center'),
	html.Hr(),
	INPUT_DIV,
	dbc.Row(
		dbc.Col(
			dbc.Tab(OPTIMIZE_DIV, label='Optimize Lineup'), 
			width=8), 
		justify='center'),
	dbc.Row(
		dbc.Col(
			dbc.Tabs([
				dbc.Tab(SIMULATE_BRANCH_DIV, label='Single Simulation'),
				dbc.Tab(SIMULATE_MULTI_DIV, label='Multi-Branch Simulation')
				]),
			width=8
			),
		justify='center'
		),
])

# Info / Credits div.
# ADD: INFO ABOUT MODEL, DATA SHOUTOUT, LINK TO REPO
CREDITS_DIV = html.Div(id='credits', children=[
	dbc.Row(
		dbc.Col(
			html.P("""
				Claude Fried, 2021.
				Credits etc.
				"""),
			width='auto'),
		justify='end'
		)
])

################################################################################
################################################################################
# App ##########################################################################
################################################################################
################################################################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])
app.title = 'Neural Networks in MLB'
server = app.server

app.layout = html.Div(
	id='main_div', 
	children=[
		HEADING_DIV,
		html.Hr(),
		html.Div(id='data-output', style={'display': 'none'}), #hidden with data
		CREDITS_DIV,
		html.Div(id='test-output'),
		],
	style={'width': '90%', 'margin': 'auto', 'padding': '30px'})

################################################################################
################################################################################
# Callbacks ####################################################################
################################################################################
################################################################################
@app.callback(
	Output('data-output', 'children'), 
	[
	Input('hitter-1-input', 'value'),
	Input('hitter-2-input', 'value'),
	Input('hitter-3-input', 'value'),
	Input('hitter-4-input', 'value'),
	Input('hitter-5-input', 'value'),
	Input('hitter-6-input', 'value'),
	Input('hitter-7-input', 'value'),
	Input('hitter-8-input', 'value'),
	Input('hitter-9-input', 'value'),
	Input('pitcher-input', 'value'),
	Input('inning-number', 'value'),
	Input('inning-half', 'value'),
	Input('career-stats', 'value'),
	Input('doubleheader', 'value'),
	Input('day-night', 'value'),
	Input('temp', 'value'),
	Input('wind', 'value'),
	Input('conditions', 'value'),
	Input('precip', 'value'),
	Input('attendance', 'value'),
	])
def get_data(lineup1, lineup2, lineup3, lineup4, lineup5, 
			 lineup6, lineup7, lineup8, lineup9, pitcher_id,
			 inning_num, inning_half, use_career_stats, dblhdr_number, day_night,
			 temp, wind, field_cond, precip, attendance):
	"""
	Get live data from input forms. Save as json.dumps(data) in hidden div.
	"""

	lineup = [
		lineup1, 
		lineup2, 
		lineup3, 
		lineup4, 
		lineup5, 
		lineup6, 
		lineup7, 
		lineup8, 
		lineup9
	]
	start_inning = inning_num[0]
	num_innings = inning_num[1] - start_inning + 1
	data = dict(
		lineup=['' if x is None else x for x in lineup],
		pitcher_id='' if pitcher_id is None else pitcher_id,
		inning_num=start_inning,
		inning_half=inning_half,
		use_career_stats=True if use_career_stats else False,
		dblhdr_number=dblhdr_number,
		day_night=day_night,
		temp=temp,
		wind=wind,
		field_cond=field_cond,
		precip=precip,
		attendance=attendance,
		num_innings=num_innings
		)
	return json.dumps(data)


@app.callback(
	[Output('single-output-graph', 'figure'),
	 Output('single-output-md', 'children')], 
	[Input('submit-single', 'n_clicks')],
	[State('data-output', 'children')])
def show_results_single(n_clicks, data):
	"""
	Run single simulation after button click.
	Plot runs scored, return markdown of game details.
	"""

	if not n_clicks:
		fig = go.Figure()
		fig.update_layout(dict(
			plot_bgcolor='rgba(0, 0, 0, 0)',
			paper_bgcolor='rgba(0, 0, 0, 0)')
			)
		return fig, ''
		# raise PreventUpdate

	data = json.loads(data)
	inning_range = range(
		data['inning_num'], 
		data['inning_num'] + data['num_innings'])
	results, verbose_results = simulator.simulate_branch(**data)
	df = pd.DataFrame(
		zip(inning_range, 
			results), 
		columns=['Inning', 'Runs'])
	fig = px.bar(df, x='Inning', y='Runs', title='Simulation Results')
	fig.add_trace(
		go.Scatter(
			x=df['Inning'], 
			y=df['Runs'].cumsum(), 
			name='Total Runs Scored'
			)
		)
	fig.update_xaxes(tickvals=[str(x) for x in inning_range])
	fig.update_layout(dict(
		plot_bgcolor='rgba(0, 0, 0, 0)',
		paper_bgcolor='rgba(0, 0, 0, 0)')
		)

	md_results = ''
	for statement in verbose_results:
		if statement.startswith('With'):
			md_results += f'- {statement}\n'
		else:
			md_results += f'#### {statement}\n'
	return fig, md_results


@app.callback(
	[Output('multi-output-graph', 'figure'),
	 Output('multi-output-md', 'children')], 
	[Input('submit-multi', 'n_clicks')],
	[State('num-simulations', 'value'),
	 State('data-output', 'children')])
def show_results_simulate(n_clicks, n, data):
	"""
	Run multiple game simulations to see the lineup's output over a long 
	period of time.
	"""

	if not n_clicks:
		fig = go.Figure()
		fig.update_layout(dict(
			plot_bgcolor='rgba(0, 0, 0, 0)',
			paper_bgcolor='rgba(0, 0, 0, 0)')
			)
		return fig, ''
		# raise PreventUpdate

	df, verbose_results = simulator.simulate(n=n, **json.loads(data))
	fig = px.histogram(df, x='simulation_total')
	# fig = create_distplot(
	# 	[df['simulation_total']], 
	# 	['Runs Scored'],
	# 	bin_size=0.5)
	fig.update_layout(
		title=f'Simulation Results ({len(df)} simulations)',
		xaxis_title='Total Runs Scored per Simulation',
		yaxis_title='Percent of Games',
		)
	fig.update_layout(dict(
		plot_bgcolor='rgba(0, 0, 0, 0)',
		paper_bgcolor='rgba(0, 0, 0, 0)')
		)
	return fig, verbose_results


@app.callback(Output('optimze-interval-component', 'disabled'),
			 [Input('toggle-interval', 'checked')])
def start_interval_component(checked):
	"""
	Set interval component to run.
	While running, the `storage` class will be checked and printed at each 
	timed-interval.
	"""

	if checked:
		return False
	return True


@app.callback([Output('lineup-locks-output', 'children'),
			   Output('trying', 'children')],
			  [Input('optimze-interval-component', 'n_intervals'),
			   Input('optimze-interval-component', 'disabled')])
def show_optimize_progress(n_intervals, disabled):
	"""
	The `storage` class is checked (the data from which is updated during the
	optimizer).
	"""

	if disabled:
		return '', ''
	if not storage.locked_in and not storage.currently_trying:
		return '', ''

	locked_in_names = [
		f'{n}. {simulator.player_finder.get_player_name(hitter, verbose=False)}' 
		for n, hitter in enumerate(storage.locked_in, 1)
	]
	locks_message = '### Lineup Locks:\n' + \
					'\n'.join(locked_in_names)
	if len(locked_in_names) == 9:
		trying_message = ''
	else:
		trying_message = '### Currently Trying:\n' + \
						 '\n'.join(storage.currently_trying)
	return locks_message, trying_message


@app.callback(
	[Output('hitter-1-input', 'value'),
	 Output('hitter-2-input', 'value'),
	 Output('hitter-3-input', 'value'),
	 Output('hitter-4-input', 'value'),
	 Output('hitter-5-input', 'value'),
	 Output('hitter-6-input', 'value'),
	 Output('hitter-7-input', 'value'),
	 Output('hitter-8-input', 'value'),
	 Output('hitter-9-input', 'value'),
	 Output('loading-optimize', 'children')],
	[Input('submit-optimize', 'n_clicks')],
	[State('sims-per-order', 'value'), 
	 State('data-output', 'children')]
	)
def optimize_lineup(n_clicks, simulations_per_order, data):
	"""
	Iterate through each lineup-spot to find out who belongs where in the order.
	"""

	global storage

	if not n_clicks:
		raise PreventUpdate

	storage.locked_in = []
	storage.currently_trying = []

	# Load data
	data = json.loads(data)
	hitters = data['lineup']
	data.pop('lineup', None)
	# Find optimized hitter for each spot in order.
	for start_idx in range(9): # Top players for first 8 spots.
		_hitters = [] # tuple of stats of the simulations
		for batting_order in shuffle_lst(
			hitters, 
			start_idx, 
			masked_elements=8-len(storage.locked_in)):
			# Update `trying`
			storage.currently_trying = [
				f'{n}. {simulator.player_finder.get_player_name(hitter, verbose=False)}'
				for n, hitter in enumerate(batting_order, 1)
			]

			# Run simulations
			df, _ = simulator.simulate(
				lineup=batting_order,
				n=simulations_per_order,
				**data)

			# Append (hitter_id, total_runs, count_of_scores).
			runs = df['simulation_total'].copy()
			# Remove bottom and top 25%
			runs = runs[(runs > runs.quantile(0.25)) & 
						(runs < runs.quantile(0.75))]
			_scoring_sims = runs[runs > 0]
			_hitters.append(
				(batting_order[start_idx], 
				 runs.sum(), # sum of all the sims
				 len(_scoring_sims)) # number of sims with runs scored
			)
		# Find player with highest expected_runs_scored.
		top_hitter = sorted(
			_hitters, 
			key=lambda x: (x[1], x[2]), 
			reverse=True
			)[0][0]
		storage.locked_in.append(top_hitter)
		
		# Remove locked in players from hitters list.
		hitters = [h for h in hitters if h not in storage.locked_in]

		# Reset hitters list starting with locked in players 
		# so locked_in players won't be iterated over.
		hitters = storage.locked_in + hitters

	h1, h2, h3, h4, h5, h6, h7, h8, h9 = hitters
	success_msg = '\n#### Lineup Sorted Sucessfully\n'

	# set lineup inputs to optimized lineup
	return h1, h2, h3, h4, h5, h6, h7, h8, h9, success_msg


if __name__ == '__main__':
	app.run_server(debug=True)
