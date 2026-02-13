import csv
import sys
import os

def detect_type(value):
    try:
        int(value)
        return "int"
    except ValueError:
        try:
            float(value)
            return "float"
        except ValueError:
            return "string"

def escape_c_string(s):
    return (
        s.replace("\\", "\\\\")
         .replace("\"", "\\\"")
         .replace("\n", "\\n")
    )

def csv_to_header(csv_path, header_path,
                  row_struct,
                  table_struct,
                  table_name):
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError("CSV must have header + data")

    headers = rows[0]
    data = rows[1]
    data_rows = rows[1:]

    col_types = [detect_type(v) for v in data]

    guard = os.path.basename(header_path).upper().replace(".", "_")

    with open(header_path, "w", encoding="utf-8") as h:
        h.write(f"#ifndef {guard}\n#define {guard}\n\n")
        h.write("// Auto-generated from CSV\n\n")

        # Row struct
        h.write("typedef struct {\n")
        for name, t in zip(headers, col_types):
            if t == "int":
                h.write(f"    int {name};\n")
            elif t == "float":
                h.write(f"    float {name};\n")
            else:
                h.write(f"    const char *{name};\n")
        h.write(f"}} {row_struct};\n\n")

        # Table struct
        h.write("typedef struct {\n")
        h.write(f"    const {row_struct} *data;\n")
        h.write("    unsigned int count;\n")
        h.write(f"}} {table_struct};\n\n")

        rows_name = f"{table_name}_rows"

        # Data
        h.write(f"static const {row_struct} {rows_name}[] = {{\n")
        for row in data_rows:
            values = []
            for value, t in zip(row, col_types):
                if t == "string":
                    values.append(f"\"{escape_c_string(value)}\"")
                else:
                    values.append(value)
            h.write("    { " + ", ".join(values) + " },\n")
        h.write("};\n\n")

        # Table instance
        h.write(f"static const {table_struct} {table_name} = {{\n")
        h.write(f"    .data = {rows_name},\n")
        h.write(f"    .count = {len(data_rows)}\n")
        h.write("};\n\n")

        h.write("#endif\n")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python csv_to_h.py input.csv output.h RowStruct TableStruct table_name")
        sys.exit(1)

    csv_to_header(
        csv_path=sys.argv[1],
        header_path=sys.argv[2],
        row_struct=sys.argv[3],
        table_struct=sys.argv[4],
        table_name=sys.argv[5],
    )
