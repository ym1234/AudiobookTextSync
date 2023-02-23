#!/usr/bin/env python3

import argparse
import csv
import requests
import os
import tempfile
import json

ANKI_CONNECT_URL = ''


def deep_copy(d):
    return json.loads(json.dumps(d))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Import a local or remote CSV file into Anki"
    )

    parser.add_argument("-p", "--path", help="the path of the local CSV file")
    parser.add_argument("-u", "--url", help="the URL of the remote CSV file")

    parser.add_argument(
        "--no-anki-connect",
        help="write notes directly to Anki DB without using AnkiConnect",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--col",
        help="the path to the .anki2 collection (only when using --no-anki-connect)",
    )
    parser.add_argument(
        "--allow-html",
        help="render HTML instead of treating it as plaintext (only when using --no-anki-connect)",
        action="store_true",
    )
    parser.add_argument(
        "--skip-header",
        help="skip first row of CSV (only when using --no-anki-connect)",
        action="store_true",
    )
    parser.add_argument(
        "--mapping",
        help="The file that defines how to map your note to your tsv",
        action="store",
        required=True
    )
    parser.add_argument(
        "--name",
        help="The name of the file to add to the deck name - so if you have a deckName in your mapping of Default, and pass in hello for the name you would get Default::hello",
        action="store",
        required=True
    )
    parser.add_argument(
        "--anki-connect-url",
        help="The url in AnkiConnect, 'localhost:8755' or $(hostname).local:8765. If using WSL2, AnkiConnect in Anki's addon config shows webBindAddress: 0.0.0.0, webBindPort: 8765 and you should use `export ANKICONNECT=http://$(hostname).local:8765` in an env variable to talk to Anki in windows from your Linux Distro. https://github.com/microsoft/WSL/issues/5211#issuecomment-751945803",
        action="store",
        required=True
    )

    return parser.parse_args()


def validate_args(args):
    if args.path and args.url:
        print("[E] Only one of --path and --url can be supplied")
        exit(1)

    if not (args.path or args.url):
        print("[E] You must specify either --path or --url")
        exit(1)

    if args.no_anki_connect:
        if not args.col:
            print("[E] --col is required when using --no-anki-connect")
            exit(1)
    else:
        if args.skip_header:
            print("[E] --skip-header is only supported with --no-anki-connect")
            exit(1)
        elif args.allow_html:
            print(
                "[E] --allow-html is only supported with --no-anki-connect, "
                "when using AnkiConnect HTML is always enabled"
            )
            exit(1)
        elif args.col:
            print("[E] --col is only supported with --no-anki-connect")
            exit(1)


def parse_ac_response(response):
    if len(response) != 2:
        raise Exception("response has an unexpected number of fields")
    if "error" not in response:
        raise Exception("response is missing required error field")
    if "result" not in response:
        raise Exception("response is missing required result field")
    if response["error"] is not None:
        raise Exception(response["error"])
    return response["result"]


def create_ac_payload(action, **params):
    return {"action": action, "params": params, "version": 6}


def invoke_ac(action, **params):
    requestJson = create_ac_payload(action, **params)
    # try:
    response = requests.post(ANKI_CONNECT_URL, json=requestJson).json()
    # except requests.exceptions.ConnectionError:
    #     print("[E] Failed to connect to AnkiConnect, make sure Anki is running")
    #     exit(1)

    return parse_ac_response(response)


def get_fields(note_type):
    return invoke_ac("modelFieldNames", modelName=note_type)


def map_fields_to_note(row, field_mappings):
    fields = {}
    for field_name, csv_index in field_mappings.items():
        fields[field_name] = row[csv_index - 1]

    return fields


def csv_to_ac_notes(csv_path, note_template, field_mappings):
    notes = []
    # model_fields = get_fields(note_type)

    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        for i, row in enumerate(reader):
            note = deep_copy(note_template)
            # Doing the one field that needs to be deep merged manually
            note["fields"] = deep_copy(note_template["fields"] | map_fields_to_note(row, field_mappings))
            notes.append(note)

    return notes


def set_empty_fields(note_template):
    for k, v in note_template['fields'].items():
        if v == '':
            note_template['fields'][k] = 'empty'
    return note_template


def replace_empty_fields(note_template):
    for k, v in note_template['fields'].items():
        if v == 'empty':
            note_template['fields'][k] = ''
    return note_template


def set_empty(empty_fields, ids):
    print("[+] Emptying any fields left empty intentionally")
    actions = []
    for id in ids:
        note = {
            "id": id,
            "fields": empty_fields
        }
        actions.append(create_ac_payload('updateNoteFields', note=note))
    return invoke_ac('multi', actions=actions)

def send_to_anki_connect(csv_path, note_template, field_mappings):
    print("[+] Preparing new notes")
    empty_fields_note_template = set_empty_fields(note_template)
    notes = csv_to_ac_notes(csv_path, empty_fields_note_template, field_mappings)
    notes_response = invoke_ac("addNotes", notes=notes)
    successes = [x for x in notes_response if x is not None]
    failures = len(notes) - len(successes)

    empty = set_empty(replace_empty_fields(empty_fields_note_template)['fields'], successes)

    print("[+] Created {} new notes".format(len(successes)))
    if failures:
        print(f"Failed to create: {failures}. Some cards failed. Maybe their primary field was left empty in the mapping?")

def get_mapping(mapping_path):
    with open(mapping_path) as f:
        mapping = json.load(f)
        print(f"Reading mapping: {mapping}")
    return mapping


def parse_mapping(mapping):
    note_template = deep_copy(mapping)
    field_mapping = {}

    for k, v, in note_template['fields'].items():
        if isinstance(v, int):
            field_mapping[k] = v
    for k, v in field_mapping.items():
        del note_template['fields'][k]
    return note_template, field_mapping


def create_deck(deck, name):
    deck_name = f"{deck}::{name}"
    invoke_ac("createDeck", deck=deck_name)
    print(f"Deck to be used: {deck_name}")
    return deck_name


def main():
    args = parse_arguments()
    validate_args(args)
    global ANKI_CONNECT_URL
    ANKI_CONNECT_URL = args.anki_connect_url
    print(ANKI_CONNECT_URL)
    if args.path:
        # Use an existing CSV file. We convert this to an absolute path because
        # CWD might change later
        csv_path = os.path.abspath(args.path)
    else:
        assert False  # Should never reach here

    # field_mappings = [(args.expression_index, args.expression_field), (args.audio_index, args.audio_field)]
    mapping = get_mapping(args.mapping)

    deck_name = create_deck(mapping["deckName"], args.name)
    note_template, field_mappings = parse_mapping(mapping)
    note_template['deckName'] = deck_name
    send_to_anki_connect(csv_path, note_template, field_mappings)


main()
