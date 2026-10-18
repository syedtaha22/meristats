/*
  meristats, a testing and results framework for the Meridian chess engine
  Copyright (C) 2026 Syed Taha

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/* meristats site script: table search, filters and sorting, and the details of a run. */
(function () {
  "use strict";

  function all(root, selector) {
    return Array.prototype.slice.call(root.querySelectorAll(selector));
  }

  var tables = [];
  var SIZE_KEY = "meristats-page-size";
  var CHEVRON = {
    prev: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" ' +
      'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M15 6l-6 6 6 6"/></svg>',
    next: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" ' +
      'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M9 6l6 6-6 6"/></svg>'
  };

  function storedSize() {
    try {
      return parseInt(localStorage.getItem(SIZE_KEY), 10) || 10;
    } catch (e) {
      return 10;
    }
  }

  function storeSize(size) {
    try { localStorage.setItem(SIZE_KEY, String(size)); } catch (e) {}
  }

  function showRunDetails() {
    var shown = {};

    all(document, "[data-table] tbody tr").forEach(function (row) {
      if (!row.hidden) {
        shown[row.getAttribute("data-run")] = true;
      }
    });

    all(document, "details.run").forEach(function (details) {
      details.hidden = !shown[details.getAttribute("data-run")];
    });
  }

  function openRun(id, smooth) {
    var element = id ? document.getElementById(id) : null;

    if (!element || element.tagName !== "DETAILS") {
      return false;
    }

    tables.forEach(function (table) { table.reveal(id); });
    element.hidden = false;
    element.open = true;
    element.classList.remove("flash");
    void element.offsetWidth;
    element.classList.add("flash");
    element.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
    return true;
  }

  function setupTable(panel) {
    var body = panel.querySelector("tbody");
    var rows = all(panel, "tbody tr");
    var input = panel.querySelector("input[type=search]");
    var count = panel.querySelector(".count");
    var empty = panel.querySelector(".empty");
    var pager = panel.querySelector(".pager");
    var select = pager.querySelector("select");
    var range = pager.querySelector(".range");
    var nav = pager.querySelector(".pages");
    var filters = { grade: "", group: "" };
    var matched = rows.slice();
    var size = storedSize();
    var page = 1;

    if (!all(select, "option").some(function (option) { return +option.value === size; })) {
      size = 10;
    }

    select.value = String(size);

    function button(label, target, current, disabled, step) {
      var element = document.createElement("button");

      element.type = "button";
      element.className = "pg" + (current ? " on" : "") + (step ? " step" : "");
      element.disabled = !!disabled;

      if (step) {
        element.setAttribute("aria-label", label === "prev" ? "Previous page" : "Next page");
        element.innerHTML = CHEVRON[label];
      } else {
        element.textContent = label;
      }

      if (current) {
        element.setAttribute("aria-current", "page");
      }

      element.addEventListener("click", function () {
        page = target;
        render();
      });
      return element;
    }

    function numbers(pages) {
      var shown = [];
      var last = 0;

      for (var number = 1; number <= pages; number++) {
        if (number === 1 || number === pages || Math.abs(number - page) <= 1) {
          shown.push(number);
        }
      }

      return shown.reduce(function (list, number) {
        if (number - last > 1) {
          list.push(0);
        }

        list.push(number);
        last = number;
        return list;
      }, []);
    }

    function render() {
      var pages = Math.max(1, Math.ceil(matched.length / size));
      var first;

      page = Math.min(Math.max(page, 1), pages);
      first = (page - 1) * size;

      rows.forEach(function (row) { row.hidden = true; });
      matched.slice(first, first + size).forEach(function (row) { row.hidden = false; });

      count.textContent = matched.length + " of " + rows.length;
      empty.hidden = matched.length !== 0;
      pager.hidden = rows.length <= 10;
      range.textContent = matched.length
        ? (first + 1) + " to " + Math.min(first + size, matched.length) + " of " + matched.length
        : "0 of 0";
      nav.textContent = "";

      if (pages > 1) {
        nav.appendChild(button("prev", page - 1, false, page === 1, true));
        numbers(pages).forEach(function (number) {
          if (number === 0) {
            var gap = document.createElement("span");

            gap.className = "gap";
            gap.textContent = "\u2026";
            nav.appendChild(gap);
          } else {
            nav.appendChild(button(String(number), number, number === page, false));
          }
        });
        nav.appendChild(button("next", page + 1, false, page === pages, true));
      }

      showRunDetails();
    }

    function filter() {
      var terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);

      matched = rows.filter(function (row) {
        var text = row.getAttribute("data-search");
        var match = terms.every(function (term) { return text.indexOf(term) !== -1; });

        match = match && (!filters.grade || row.getAttribute("data-grade") === filters.grade);
        return match && (!filters.group || row.getAttribute("data-group") === filters.group);
      });
      page = 1;
      render();
    }

    function clear() {
      input.value = "";
      filters.grade = "";
      filters.group = "";
      all(panel, ".chips").forEach(function (chips) {
        all(chips, ".chip").forEach(function (chip, index) {
          chip.classList.toggle("on", index === 0);
        });
      });
      matched = rows.slice();
    }

    input.addEventListener("input", filter);

    select.addEventListener("change", function () {
      size = parseInt(select.value, 10);
      storeSize(size);
      tables.forEach(function (table) { table.resize(size); });
    });

    all(panel, ".chips").forEach(function (chips) {
      var kind = chips.getAttribute("data-filter");

      chips.addEventListener("click", function (event) {
        var chip = event.target.closest(".chip");

        if (!chip) {
          return;
        }

        all(chips, ".chip").forEach(function (other) { other.classList.remove("on"); });
        chip.classList.add("on");
        filters[kind] = chip.getAttribute("data-value");
        filter();
      });
    });

    var headers = all(panel, "thead th");
    var original = rows.slice();
    var sorter = panel.querySelector(".sort select");

    function sortBy(index, direction) {
      headers.forEach(function (other) { other.removeAttribute("aria-sort"); });

      if (index < 0) {
        rows = original.slice();
        sorter.value = "";
      } else {
        var numeric = headers[index].getAttribute("data-sort") === "num";
        var sign = direction === "desc" ? -1 : 1;

        headers[index].setAttribute("aria-sort", direction === "desc" ? "descending" : "ascending");
        sorter.value = index + ":" + direction;

        rows = original.slice().sort(function (a, b) {
          var left = a.children[index].getAttribute("data-sort");
          var right = b.children[index].getAttribute("data-sort");

          if (numeric) {
            left = parseFloat(left) || 0;
            right = parseFloat(right) || 0;
          } else {
            left = left.toLowerCase();
            right = right.toLowerCase();
          }

          return sign * (left < right ? -1 : left > right ? 1 : 0);
        });
      }

      rows.forEach(function (row) { body.appendChild(row); });
      matched = rows.filter(function (row) { return matched.indexOf(row) !== -1; });
      page = 1;
      render();
    }

    headers.forEach(function (header, index) {
      header.addEventListener("click", function () {
        sortBy(index, header.getAttribute("aria-sort") === "ascending" ? "desc" : "asc");
      });
    });

    sorter.addEventListener("change", function () {
      var parts = sorter.value.split(":");

      sortBy(sorter.value ? parseInt(parts[0], 10) : -1, parts[1]);
    });

    tables.push({
      resize: function (value) {
        size = value;
        select.value = String(value);
        page = 1;
        render();
      },
      reveal: function (id) {
        var found = rows.filter(function (row) { return row.getAttribute("data-run") === id; });

        if (!found.length) {
          return;
        }

        if (matched.indexOf(found[0]) === -1) {
          clear();
        }

        page = Math.floor(matched.indexOf(found[0]) / size) + 1;
        render();
      }
    });

    render();
  }

  all(document, "[data-table]").forEach(setupTable);

  document.addEventListener("click", function (event) {
    var link = event.target.closest("a[href^='#run-']");

    if (link && openRun(link.getAttribute("href").slice(1), true)) {
      event.preventDefault();

      if (window.history && history.replaceState) {
        history.replaceState(null, "", link.getAttribute("href"));
      }
    }
  });

  window.addEventListener("hashchange", function () { openRun(location.hash.slice(1), true); });

  if (location.hash) {
    openRun(location.hash.slice(1), false);
  }

  var search = document.getElementById("site-search");
  var hits = document.getElementById("site-hits");
  var index = document.getElementById("site-index");

  if (search && hits && index) {
    var items = JSON.parse(index.textContent);

    search.addEventListener("input", function () {
      var terms = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);

      hits.textContent = "";

      if (!terms.length) {
        hits.hidden = true;
        return;
      }

      var found = items.filter(function (item) {
        var text = (item.label + " " + item.page + " " + item.text + " " + item.grade).toLowerCase();
        return terms.every(function (term) { return text.indexOf(term) !== -1; });
      });

      found.slice(0, 12).forEach(function (item) {
        var li = document.createElement("li");
        var a = document.createElement("a");
        var what = document.createElement("span");
        var meta = document.createElement("span");
        var where = document.createElement("span");

        a.href = item.href;
        what.className = "what";
        what.textContent = item.label;
        meta.className = "where";
        meta.textContent = item.text;
        where.className = "where";
        where.textContent = item.page;
        meta.style.marginLeft = "auto";
        where.style.marginLeft = "0";
        a.appendChild(what);
        a.appendChild(meta);
        a.appendChild(where);
        li.appendChild(a);
        hits.appendChild(li);
      });

      if (!found.length) {
        var none = document.createElement("li");
        none.innerHTML = "<a><span class='where'>Nothing matches.</span></a>";
        hits.appendChild(none);
      }

      hits.hidden = false;
    });
  }
})();
