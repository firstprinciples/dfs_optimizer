import pandas as pd
import datetime as dt
import time

from draft_kings.data import Sport
from draft_kings.client import contests, draftables

def get_today():
    return dt.date.fromtimestamp(time.time())

def get_contests(date=None, min_fee=0.25, max_fee=101):
    the_contests = contests(sport=Sport.NBA)
    df_contests = pd.DataFrame(the_contests['contests'])
    entries = list(df_contests['entries'].values)
    df_contests = pd.concat((df_contests,pd.DataFrame(entries)), axis=1)
    df_contests = df_contests[df_contests['fee']<=max_fee]
    df_contests = df_contests[df_contests['fee']>=min_fee]
    if date is not None:
        inds = []
        for ind in df_contests.index:
            if df_contests.loc[ind, 'starts_at'].date() == date:
                inds += [ind]
        
        df_contests = df_contests.loc[inds]
        df_contests.reset_index(0, inplace=True, drop=True)
    return df_contests

def get_players(draft_group_id):
    draft_players = draftables(draft_group_id)
    df_players = pd.DataFrame(draft_players['draftables'])
    names = list(df_players['names'].values)
    df_names = pd.DataFrame(names)
    df_players = pd.concat((df_players, df_names),axis=1)
    competitions = list(df_players['competition'].values)
    df_comps = pd.DataFrame(competitions)
    df_comps.columns = ['contest id', 'contest names', 'start time']
    df_players = pd.concat((df_players, df_comps),axis=1)
    df_players.reset_index(0, drop=True, inplace=True)
    map_cols = {'display' : 'Name', 'id' : 'ID', 'position' : 'Position',
        'news_status' : 'Status', 'salary' : 'Salary', 
        'contest names' : 'Game', 'start time' : 'Time',
        'team_abbreviation' : 'TeamAbbrev'}
    df_players.rename(columns=map_cols, inplace=True)
    return df_players[list(map_cols.values())]

def read_date(x):
    _, date, time, tz = x.split(' ')
    return date + ' ' + time

def read_game(x):
    game = x.split(' ')[0]
    away, home = game.split('@')
    return away + ' @ ' + home

def get_players_from_salaries(path):
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Game Info'].apply(read_date))
    df['Game'] = df['Game Info'].apply(read_game)
    cols_out = ['Name', 'ID', 'Position', 'Salary', 'Game', 'Time', 'TeamAbbrev']
    optional_cols = ['projections', 'std', 'min_exp', 'max_exp']
    for col in optional_cols:
        if col in df.columns:
            cols_out += [col]
    return df[cols_out]

class EntriesHandler:

    ENTRIES_COLS = ['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee']
    POSITION_COLS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    
    def __init__(self, entries_path, df, read_lineups=False):
        self.entries_path = entries_path
        self.df = df
        self.df_sheet_lineups = pd.DataFrame(columns=self.POSITION_COLS)
        self._read_entries()
        self._make_id_map_dicts()
        if read_lineups:
            self._read_lineups()

    def _read_entries(self):
        df_entries = pd.read_csv(
            self.entries_path, usecols=self.ENTRIES_COLS + self.POSITION_COLS)
        df_entries.dropna(axis=0, how='all', inplace=True)
        df_entries.fillna(0, inplace=True)
        df_entries[['Entry ID', 'Contest ID']] = \
            df_entries[['Entry ID', 'Contest ID']].astype(int)
        self.df_entries = df_entries

    def _make_id_map_dicts(self):
        self.name_to_id_dict = self.df['ID'].to_dict()
        self.locked_to_id_dict = {
            name + ' (LOCKED)' : id_ \
                for (name, id_) in self.name_to_id_dict.items()}
        self.id_to_name_dict = {
            id_ : name for (name, id_) in self.name_to_id_dict.items()}
        self.str_to_name_dict = {
            name + ' (' + str(id_) + ')' : name \
                for (name, id_) in self.name_to_id_dict.items()}
        self.str_id_to_name_dict = {
            str(id_) : name for (name, id_) in self.name_to_id_dict.items()}
        self.locked_to_name_dict = {
            str_ + ' (LOCKED)' : name + ' (LOCKED)' \
                for (str_, name) in self.str_to_name_dict.items()}
        self.entry_map = dict(
            list(self.id_to_name_dict.items()) + \
                list(self.str_to_name_dict.items()) + \
                    list(self.str_id_to_name_dict.items()) + \
                        list(self.locked_to_name_dict.items()))
        self.exit_map = dict(
            list(self.name_to_id_dict.items()) + \
                list(self.locked_to_id_dict.items()))

    def _read_lineups(self):
        df_sheet_lineups = pd.DataFrame(columns=self.POSITION_COLS)
        for col in self.POSITION_COLS:
            df_sheet_lineups[col] = self.df_entries[col].map(self.entry_map)
        df_sheet_lineups.dropna(axis=0, how='any', inplace=True)
        df_sheet_lineups.reset_index(0, drop=True, inplace=True)
        self.df_sheet_lineups = df_sheet_lineups

    def add_lineups_to_entries(self, df_lineups, version=2):
        self.df_sheet_lineups = pd.concat((self.df_sheet_lineups, df_lineups))
        self.df_sheet_lineups.drop_duplicates(inplace=True)
        drop = max(0, len(self.df_sheet_lineups) - len(self.df_entries))
        if drop > 0:
            print('Dropping {} lineups'.format(drop))
            self.df_sheet_lineups = self.df_sheet_lineups.iloc[:-drop]

        self.df_sheet_lineups.reset_index(0, inplace=True, drop=True)
        self.df_entries = pd.concat(
            (self.df_entries[self.ENTRIES_COLS], self.df_sheet_lineups), axis=1)
        self._write_entries_to_csv(version=version)

    def _write_entries_to_csv(self, version=2):
        df_entries_out = pd.DataFrame(columns=self.POSITION_COLS)
        for col in self.POSITION_COLS:
            df_entries_out[col] = self.df_entries[col].map(self.exit_map)
        
        df_entries_out = pd.concat(
            (self.df_entries[self.ENTRIES_COLS], df_entries_out), axis=1)
        df_entries_out.to_csv(
            self.entries_path[:-4] + \
                '_v' + str(version) + '.csv', index=False)