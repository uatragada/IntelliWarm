## Important Notes

- The test suite is clean, but it still emits Flask/Werkzeug/`itsdangerous` deprecation warnings from the current dependency stack. They are not breaking behavior today, but upgrading that stack should be scheduled before broader framework work.
