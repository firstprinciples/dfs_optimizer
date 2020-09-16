import pandas as pd
import numpy as np

from draft_kings.data import Sport
from draft_kings.client import contests, draftables


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

def fix_exceptions(first_name, last_name):
    playerid_fix = None
    if last_name == 'Osman':
        playerid_fix = 'osmande01'
    if last_name == 'Morris' and first_name == 'Markieff':
        playerid_fix = 'morrima02'
    elif ('Morris' in last_name) and (first_name == 'Marcus'):
        playerid_fix = 'morrima03'
    if last_name == 'Bridges' and first_name == 'Mikal':
        playerid_fix = 'bridgmi01'
    elif last_name == 'Bridges' and first_name == 'Miles':
        playerid_fix = 'bridgmi02'
    if last_name == 'Millsap':
        playerid_fix = 'millspa01'
    elif last_name == 'Mills':
        playerid_fix = 'millspa02'
    if last_name == 'Bogdanovic' and first_name == 'Bogdan':
        playerid_fix = 'bogdabo01'
    elif last_name == 'Bogdanovic' and first_name == 'Bojan':
        playerid_fix = 'bogdabo02'
    if last_name == 'Kleber':
        playerid_fix = 'klebima01'
    if last_name == 'Harkless':
        playerid_fix = 'harklma01'
    if last_name == 'Barea':
        playerid_fix = 'bareajo01'
    if last_name == 'Ntilikina':
        playerid_fix = 'ntilila01'
    if last_name == 'Hernangómez':
        playerid_fix = 'hernawi01'
    if last_name == 'Capela':
        playerid_fix = 'capelca01'
    if last_name == 'Green' and first_name == 'JaMychal':
        playerid_fix = 'greenja01'
    elif last_name == 'Green' and first_name == 'Javonte':
        playerid_fix = 'greenja02'
    return playerid_fix

def get_players(
    draft_group_id, 
    exceptions = ['osmance','morrima','bridgmi','millspa','bogdabo','klebema','harklmo','bareajj','ntilifr','hernagu','capelcl','greenja']):

    draft_players = draftables(draft_group_id)
    df_players = pd.DataFrame(draft_players['draftables'])
    names = list(df_players['names'].values)
    df_names = pd.DataFrame(names)
    df_players = pd.concat((df_players, df_names),axis=1)
    competitions = list(df_players['competition'].values)
    df_comps = pd.DataFrame(competitions)
    df_comps.columns = ['contest id', 'contest names', 'start time']
    df_players = pd.concat((df_players,df_comps),axis=1)
    first_names = list(df_players['first'].values)
    last_names = list(df_players['last'].values)
    first_last = [first_names[i]+'_'+last_names[i] for i in range(len(first_names))]
    playerid = []
    for i in range(len(first_names)):
        playerid += [last_names[i].replace("'",'').replace('.','').replace('-','')[:5].lower()+first_names[i].replace("'",'').replace('.','').replace('-','')[:2].lower()]
        if playerid[-1] in exceptions:
            playerid[-1] = fix_exceptions(first_names[i],last_names[i])

    df_players['first_last'] = first_last
    df_players['playerid'] = playerid
    df_players = df_players.drop_duplicates('first_last')
    df_players.reset_index(0, drop=True, inplace=True)
    return df_players