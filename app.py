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

with open(r"player_options.json", "r") as read_file:
    player_options = json.load(read_file)


def player_dropdown(_id, player_id=''):
	"""Returns a dcc.Dropdown of players."""
	return dcc.Dropdown(
		id=_id,
		# options=player_options, 
		placeholder='Last, First: Debut-Year.',
		value=player_id
 	)


asg97 = [
	'andeb001',
	'rodra001',
	'grifk002',
	'martt002',
	'marte001',
	'oneip001',
	'ripkc001',
	'rodri001',
	'alomr001'
]
maddux = 'maddg002'

lineup_selector = dbc.Col([
	dbc.Row(html.H4('Batting Order'))] + [
	dbc.Row(
		dbc.Col(player_dropdown(f'hitter-{n}-input', player_id)), 
		justify='center')
	for n, player_id in zip(range(1, 10), asg97)],
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
				dbc.Label('Number of Simulations (max: 25)'),
				dbc.Input(
					id='num-simulations', 
					type='number', value=25, min=2, max=25, step=1)
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
		html.Div(id='multi-output-md')
		])
])

# Optimizer div.
def create_optimize_div():
	"""Return a clear Optimize Div."""

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
						type='number', value=25, min=2, max=100, step=1)]),
				dbc.Button(
					'Optimize Given Lineup',
					id='submit-optimize',
					color='primary'
					),
				dbc.Button(
					'Reset',
					id='clear',
					color='secondary'
					)],
				width='auto'),
			justify='center'
			),
		html.Div(children=[
			dcc.Interval(id='interval', disabled=True),
			html.Div(id='hidden-all-hitters'),
			html.Div(id='hidden-locked-in'),
			html.Div(id='hidden-shuffled-lineups'),
			html.Div(id='hidden-current-try'),
			html.Div(id='hidden-lineup-results'),
			html.Div(id='hidden-final-output')
			], 
			style={'display': 'none'}),
		html.Br(),
		dbc.Row(
			dbc.Col(dcc.Markdown(id='working-header'), width='auto'), 
			justify='center'),
		dbc.Row([
			dbc.Col(dcc.Markdown(id='lineup-locks-output'), width='auto'),
			dbc.Col(dcc.Markdown(id='trying'), width='auto'),
			dbc.Col(dcc.Markdown(id='final-output'), width='auto')
			],
			justify='center')
		])
	return OPTIMIZE_DIV

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
			html.H3('Simulating Gameplay using Machine Learning'),
			width='auto'),
		justify='center'),
	dbc.Row(
		dbc.Col(
			html.P("""
				Since its inception in the mid-1800s, baseball has continued to 
				be a hallmark of American culture. With player and game data 
				going back well over 100 years, it's a ripe domain to study.
				"""),
			width=8
			),
		justify='center'),
	dbc.Row(
		dbc.Col(
			html.P("""
				This app uses an SGD Logistic Regression model (built in 
				scikit-learn) and "event" data that I've collected and 
				engineered from Retrosheet.org.
				My goal was to make a model that could predict the outcome of an
				"event" (at-bat) during a game. There are 10 outcomes of an 
				event, such as ['single', 'strikeout', 'home-run', 'walk', etc].
				The games are reconstructed by taking the results of each play 
				and inputting them into the game-state. Amazingly, without any 
				knowledge of what "runs-scored" are, and without any baserunning 
				considerations, the model's simulations are very good 
				representations of reality as validated by season-simulations of 
				past years).
				"""),
			width=8
			),
		justify='center'),
	dbc.Row(
		dbc.Col(
			html.P(""" 
				There are two aspects to factor in: player-data and game-data.
				Every player, both pitchers and hitters, since 1950 has been 
				accounted for as well as their at-bat to at-bat stats. The 
				hitter vs pitcher dynamic is a significant factor in how the 
				model predicts the outcomes. There is also game data, such as 
				temperature and stadium attendance, which the model uses to 
				influence its predictions. Finally, there is dynamic data of 
				prior outcomes within in an inning. For example, if the 
				previous 2 hitters just walked, how does that impact what is
				likely to happen next?
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
				the highest **expected runs scored** over each iteration through 
				the search. 
				\n > *The searching algorithm is in no way exhaustive, 
				but returns some very interesting results. By increasing 
				**Simulations per Order Shuffle**, results will be more 
				consistent.*
				\n\n- **Single Game Simulation:**
				\n > Run a detailed simulation where every play in a 
				simulated game is shown.
				\n\n- **Multi-Game Simulation:**
				\n > Run multiple simulations where a lineup faces a pitcher. 
				This is useful to see how a lineup is expected to perform over a
				long stretch of time.
				"""),
			width=8
			),
		justify='center'),
	dbc.Row(
		dbc.Col(
			dcc.Markdown("""
				***Note:*** *By default, the 1997 All Star Game starters were 
				selected for the lineup (AL batting, NL pitching). However, have 
				fun mixing and matching players from different teams, different 
				leagues, or different eras!*
				"""),
			width='auto'
			),
		justify='center'),
	html.Hr(),
	INPUT_DIV,
	dbc.Row(
		dbc.Col(
			dbc.Row(
				create_optimize_div(), 
				id='optimize-tab'
				), 
			width='auto'), 
		justify='center'),
	html.Br(),
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
			dcc.Markdown("""Claude Fried, 2021.
				\nCapstone project for Flatiron School's Data Science Bootcamp.
				\n<a href="https://github.com/cwf231/dsc_capstone">github</a>
				> The information used here was obtained free of
				> charge from and is copyrighted by Retrosheet.  Interested
				> parties may contact Retrosheet at "www.retrosheet.org".
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
app.title = 'Machine Learning in MLB'
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
# Optimizer Callbacks ##########################################################
################################################################################
################################################################################


@app.callback(
	Output('hidden-all-hitters', 'children'),
	Input('submit-optimize', 'n_clicks'),
	State('data-output', 'children')
	)
def populate_hitters(n_clicks, data):
	"""Hidden current-hitters"""

	if not n_clicks:
		raise PreventUpdate
	data = json.loads(data)
	lineup = data.get('lineup', [])
	return json.dumps(lineup)


@app.callback(
	Output('hidden-shuffled-lineups', 'children'),
	Input('interval', 'n_intervals'),
	State('hidden-all-hitters', 'children'),
	State('hidden-locked-in', 'children')
	)
def shuffle_lineups(n_intervals, all_hitters, locked_in):
	if not all_hitters:
		raise PreventUpdate
	all_hitters = json.loads(all_hitters)
	locked_in = [] if locked_in is None else json.loads(locked_in)

	order = locked_in + [h for h in all_hitters if h not in locked_in]
	start_idx = len(locked_in)

	shuffled = shuffle_lst(
		order, 
		start_idx=start_idx, 
		masked_elements=len(all_hitters)-1-start_idx
		)
	return json.dumps(shuffled)


@app.callback(
	Output('hidden-current-try', 'children'),
	Input('interval', 'n_intervals'),
	State('hidden-shuffled-lineups', 'children'),
	State('hidden-lineup-results', 'children'),
	State('hidden-current-try', 'children'),
	State('hidden-locked-in', 'children')
	)
def get_current_try(n_intervals, 
					shuffled_lineups, 
					lineup_results, 
					current_try, 
					locked_in):
	if shuffled_lineups is None:
		raise PreventUpdate
	shuffled_lineups = json.loads(shuffled_lineups)
	lineup_results = [] if not lineup_results else json.loads(lineup_results)
	locked_in = [] if not locked_in else json.loads(locked_in)
	current_try = [] if not current_try else json.loads(current_try)

	if len(lineup_results) > len(shuffled_lineups):
		lineup_results = []
	if len(lineup_results) == len(shuffled_lineups):
		raise PreventUpdate

	next_try = shuffled_lineups[len(lineup_results)]

	if current_try == next_try:
		raise PreventUpdate
	return json.dumps(next_try)


@app.callback(
	Output('hidden-lineup-results', 'children'),
	Input('hidden-current-try', 'children'),
	State('hidden-lineup-results', 'children'),
	State('hidden-locked-in', 'children'),
	State('hidden-shuffled-lineups', 'children'),
	State('sims-per-order', 'value'), 
	State('data-output', 'children')
	)
def simulate(current_try, 
			 lineup_results, 
			 locked_in, 
			 shuffled_lineups,
			 simulations_per_order,
			 data):
	if not current_try:
		raise PreventUpdate
	current_try = json.loads(current_try)
	locked_in = [] if not locked_in else json.loads(locked_in)

	# Load data
	data = json.loads(data)
	data.pop('lineup', None)
	data['num_innings'] = (len(locked_in)//3) + 1 # [1,2,3] == 1, [4,5,6] == 2, [7,8] == 3

	df, _ = simulator.simulate(
		lineup=current_try,
		n=simulations_per_order,
		**data)

	lineup_results = [] if not lineup_results else json.loads(lineup_results)
	lineups = [] if not shuffled_lineups else json.loads(shuffled_lineups)

	active_hitters = [x for x in current_try if x] # actual hitters, not control group
	h = active_hitters[-1]

	# Clear if locked in has been updated.
	if len(lineup_results) > len(lineups):
		lineup_results = []

	# Append (hitter_id, total_runs, count_of_scores).
	runs = df['simulation_total'].copy()
	# Remove bottom and top 25%
	if len(runs) > 3:
		runs = runs[(runs > runs.quantile(0.25)) & 
					(runs < runs.quantile(0.75))]
	scoring_sims = runs[runs > 0]
	total_runs = float(runs.sum()) # sum of all the sims
	num_sims_scored = float(len(scoring_sims)) # number of sims with runs scored

	lineup_results.append(
			(h, total_runs, num_sims_scored)
		)

	return json.dumps(lineup_results)


@app.callback(
	Output('hidden-locked-in', 'children'),
	Input('hidden-lineup-results', 'children'),
	State('hidden-locked-in', 'children'),
	State('hidden-all-hitters', 'children')
	)
def set_locked_in(lineup_results, locked_in, all_hitters):
	if not lineup_results:
		raise PreventUpdate
	lineup_results = [] if not lineup_results else json.loads(lineup_results)
	locked_in = [] if not locked_in else json.loads(locked_in)
	all_hitters = [] if not all_hitters else json.loads(all_hitters)

	if len(lineup_results + locked_in) != len(all_hitters):
		raise PreventUpdate

	# Find player with highest expected_runs_scored.
	top_hitter = sorted(
		lineup_results, 
		key=lambda x: (x[1], x[2]), 
		reverse=True
		)[0][0]
	locked_in.append(top_hitter)
	return json.dumps(locked_in)


@app.callback(
	Output('hidden-final-output', 'children'),
	Input('hidden-shuffled-lineups', 'children'),
	State('hidden-all-hitters', 'children')
	)
def set_final_result(shuffled_lineups, all_hitters):
	lineups = [] if not shuffled_lineups else json.loads(shuffled_lineups)
	if not lineups:
		raise PreventUpdate
	all_hitters = [] if not all_hitters else json.loads(all_hitters)
	active_hitters = [x for x in lineups[0] if x]
	if len(active_hitters) == len(all_hitters):
		return json.dumps(lineups[0])
	else:
		raise PreventUpdate


@app.callback(
	Output('interval', 'disabled'),
	Input('hidden-final-output', 'children'),
	Input('hidden-all-hitters', 'children')
	)
def reset_interval(final_output, all_hitters):
	if all_hitters and not final_output:
		return False
	else:
		return True


@app.callback(
	Output('optimize-tab', 'children'),
	Input('clear', 'n_clicks')
	)
def reset_page(n_clicks):
	if not n_clicks:
		raise PreventUpdate
	return create_optimize_div()


@app.callback(
	Output('working-header', 'children'),
	Output('lineup-locks-output', 'children'),
	Output('trying', 'children'),
	Output('final-output', 'children'),
	Input('hidden-all-hitters', 'children'),
	Input('hidden-locked-in', 'children'),
	Input('hidden-current-try', 'children'),
	Input('hidden-final-output', 'children'))
def show_optimize_progress(all_hitters, locked_in, current_try, final_output):
	"""Return pretty version of hidden divs for user."""

	WORKING = '### Working...'

	if all_hitters and not locked_in and not current_try and not final_output:
		return WORKING, '', '', ''

	if not locked_in and not current_try and not final_output:
		raise PreventUpdate

	if final_output:
		player_id_lst = json.loads(final_output)
		players = [
			f'{n}. {simulator.player_finder.get_player_name(p, verbose=False)}'
			for n, p in enumerate(player_id_lst, 1)]
		final_message = '### Optimized Lineup:\n' + \
						'\n'.join(players)
		return '### Lineup Set!', '', '', final_message

	locked_in = [] if not locked_in else json.loads(locked_in)
	current_try = [] if not current_try else json.loads(current_try)

	locked_in_names = [
		f'{n}. {simulator.player_finder.get_player_name(hitter, verbose=False)}' 
		for n, hitter in enumerate(locked_in, 1)
	]
	currently_trying_names = [
		f'{n}. {simulator.player_finder.get_player_name(hitter, verbose=False)}' 
		for n, hitter in enumerate(current_try, 1)
	]

	locks_message = '### Lineup Locks:\n' + \
					'\n'.join(locked_in_names)
	trying_message = '### Currently Trying:\n' + \
					 '\n'.join(currently_trying_names)
	return WORKING, locks_message, trying_message, ''


@app.callback(
	Output('hitter-1-input', 'value'),
	Output('hitter-2-input', 'value'),
	Output('hitter-3-input', 'value'),
	Output('hitter-4-input', 'value'),
	Output('hitter-5-input', 'value'),
	Output('hitter-6-input', 'value'),
	Output('hitter-7-input', 'value'),
	Output('hitter-8-input', 'value'),
	Output('hitter-9-input', 'value'),
	Input('hidden-final-output', 'children'))
def set_input_form_to_optimized(final_output):
	"""Set input fields to match the optimized lineup."""

	if not final_output:
		raise PreventUpdate
	return json.loads(final_output)


################################################################################
################################################################################
# Callbacks ####################################################################
################################################################################
################################################################################


@app.callback(Output('hitter-1-input', 'options'),
			 [Input('hitter-1-input', 'search_value'),
			  Input('hitter-1-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""

	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-2-input', 'options'),
			 [Input('hitter-2-input', 'search_value'),
			  Input('hitter-2-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-3-input', 'options'),
			 [Input('hitter-3-input', 'search_value'),
			  Input('hitter-3-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-4-input', 'options'),
			 [Input('hitter-4-input', 'search_value'),
			  Input('hitter-4-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-5-input', 'options'),
			 [Input('hitter-5-input', 'search_value'),
			  Input('hitter-5-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-6-input', 'options'),
			 [Input('hitter-6-input', 'search_value'),
			  Input('hitter-6-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-7-input', 'options'),
			 [Input('hitter-7-input', 'search_value'),
			  Input('hitter-7-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-8-input', 'options'),
			 [Input('hitter-8-input', 'search_value'),
			  Input('hitter-8-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('hitter-9-input', 'options'),
			 [Input('hitter-9-input', 'search_value'),
			  Input('hitter-9-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]

	
@app.callback(Output('pitcher-input', 'options'),
			 [Input('pitcher-input', 'search_value'),
			  Input('pitcher-input', 'value')])
def update_options1(search_value, value):
	"""Updates the possible dropdown options based on the given unput."""
	
	if value and not search_value:
		return [o for o in player_options if value == o['value']]
	if not search_value:
		raise PreventUpdate
	return [o for o in player_options if search_value in o['label']]


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
			 inning_num, inning_half, use_career_stats, dblhdr_number, 
			 day_night, temp, wind, field_cond, precip, attendance):
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

	df, verbose_results = simulator.simulate(n=n, **json.loads(data))
	df = df['simulation_total'].value_counts().reset_index()
	df.columns = ['Runs per Game', 'Count']
	fig = px.bar(df, x='Runs per Game', y='Count')
	fig.update_layout(
		title=f'Simulation Results ({n} simulations)',
		xaxis_title='Total Runs Scored per Simulation',
		yaxis_title='Number of Games',
		bargap=0.2
		)
	fig.update_layout(dict(
		plot_bgcolor='rgba(0, 0, 0, 0)',
		paper_bgcolor='rgba(0, 0, 0, 0)')
		)
	table = dbc.Table.from_dataframe(
		pd.DataFrame(verbose_results), 
		bordered=True)
	return fig, table


if __name__ == '__main__':
	app.run_server(debug=True)
